//! This crate reimplements most of the relevant parts of the VQF algorithm from
//! https://github.com/dlaidig/vqf/blob/f2a63375604e0b025048d181ba6a204e96ce2559/vqf/pyvqf.py Currently this is just
//! copy-pasted from the python code, but it should be made more idiomatic before actually using it. I have marked areas
//! most likely to contain bugs with ඞ
//!
//! The original code is licensed under the MIT license, so this crate is also licensed under the MIT license.

#![no_std]
#![allow(non_snake_case)]

use core::f32::consts::PI;

use nalgebra::{ArrayStorage, U2, U9};
use num_traits::Float;

type Quat = nalgebra::UnitQuaternion<f32>;
type Vec2 = nalgebra::Vector2<f32>;
type Vec3 = nalgebra::Vector3<f32>;
type Mat2x3 = nalgebra::Matrix2x3<f32>;
type Mat2x9 = nalgebra::Matrix<f32, U2, U9, ArrayStorage<f32, 2, 9>>;
type Mat2x2 = nalgebra::Matrix2<f32>;
type Mat3x3 = nalgebra::Matrix3<f32>;

const EPS: f32 = 1e-6;

pub struct VqfParameters {
	pub tauAcc: f32,
	pub tauMag: f32,
	pub motionBiasEstEnabled: bool,
	pub restBiasEstEnabled: bool,
	pub magDistRejectionEnabled: bool,
	pub biasSigmaInit: f32,
	pub biasForgettingTime: f32,
	pub biasClip: f32,
	pub biasSigmaMotion: f32,
	pub biasVerticalForgettingFactor: f32,
	pub biasSigmaRest: f32,
	pub restMinT: f32,
	pub restFilterTau: f32,
	pub restThGyr: f32,
	pub restThAcc: f32,
	pub magCurrentTau: f32,
	pub magRefTau: f32,
	pub magNormTh: f32,
	pub magDipTh: f32,
	pub magNewTime: f32,
	pub magNewFirstTime: f32,
	pub magNewMinGyr: f32,
	pub magMinUndisturbedTime: f32,
	pub magMaxRejectionTime: f32,
	pub magRejectionFactor: f32,
}

impl Default for VqfParameters {
	fn default() -> Self {
		VqfParameters {
			tauAcc: 3.0,
			tauMag: 9.0,
			motionBiasEstEnabled: true,
			restBiasEstEnabled: true,
			magDistRejectionEnabled: true,
			biasSigmaInit: 0.5,
			biasForgettingTime: 100.0,
			biasClip: 2.0,
			biasSigmaMotion: 0.1,
			biasVerticalForgettingFactor: 0.0001,
			biasSigmaRest: 0.03,
			restMinT: 1.5,
			restFilterTau: 0.5,
			restThGyr: 2.0,
			restThAcc: 0.5,
			magCurrentTau: 0.05,
			magRefTau: 20.0,
			magNormTh: 0.1,
			magDipTh: 10.0,
			magNewTime: 20.0,
			magNewFirstTime: 5.0,
			magNewMinGyr: 20.0,
			magMinUndisturbedTime: 0.5,
			magMaxRejectionTime: 60.0,
			magRejectionFactor: 2.0,
		}
	}
}

pub struct VqfState {
	pub gyrQuat: Quat,
	pub accQuat: Quat,
	pub delta: f32,
	pub restDetected: bool,
	pub magDistDetected: bool,
	pub lastAccLp: Vec3,
	pub accLpState: Mat2x3,
	pub kMagInit: f32,
	pub lastMagDisAngle: f32,
	pub lastMagCorrAngularRate: f32,
	pub bias: Vec3,
	pub biasP: Mat3x3,
	pub motionBiasEstRLpState: Mat2x9,
	pub motionBiasEstBiasLpState: Mat2x2,
	pub restLastSquaredDeviations: Vec2,
	pub restT: f32,
	pub restLastGyrLp: Vec3,
	pub restGyrLpState: Mat2x3,
	pub restLastAccLp: Vec3,
	pub restAccLpState: Mat2x3,
	pub magRefNorm: f32,
	pub magRefDip: f32,
	pub magUndisturbedT: f32,
	pub magRejectT: f32,
	pub magCandidateNorm: f32,
	pub magCandidateDip: f32,
	pub magCandidateT: f32,
	pub magNormDip: Vec2,
	pub magNormDipLpState: Mat2x2,
}

impl Default for VqfState {
	fn default() -> VqfState {
		VqfState {
			gyrQuat: Quat::identity(),
			accQuat: Quat::identity(),
			delta: 0.0,
			restDetected: false,
			magDistDetected: true,
			lastAccLp: Vec3::zeros(),
			accLpState: Mat2x3::repeat(f32::NAN),
			kMagInit: 1.0,
			lastMagDisAngle: 0.0,
			lastMagCorrAngularRate: 0.0,
			bias: Vec3::zeros(),
			biasP: Mat3x3::repeat(f32::NAN),
			motionBiasEstRLpState: Mat2x9::repeat(f32::NAN),
			motionBiasEstBiasLpState: Mat2x2::repeat(f32::NAN),
			restLastSquaredDeviations: Vec2::zeros(),
			restT: 0.0,
			restLastGyrLp: Vec3::zeros(),
			restGyrLpState: Mat2x3::repeat(f32::NAN),
			restLastAccLp: Vec3::zeros(),
			restAccLpState: Mat2x3::repeat(f32::NAN),
			magRefNorm: 0.0,
			magRefDip: 0.0,
			magUndisturbedT: 0.0,
			magRejectT: -1.0,
			magCandidateNorm: -1.0,
			magCandidateDip: 0.0,
			magCandidateT: 0.0,
			magNormDip: Vec2::zeros(),
			magNormDipLpState: Mat2x2::repeat(f32::NAN),
		}
	}
}

pub struct VQFCoefficients {
	pub gyrTs: f32,
	pub accTs: f32,
	pub magTs: f32,
	pub accLpB: Vec3,
	pub accLpA: Vec2,
	pub kMag: f32,
	pub biasP0: f32,
	pub biasV: f32,
	pub biasMotionW: f32,
	pub biasVerticalW: f32,
	pub biasRestW: f32,
	pub restGyrLpB: Vec3,
	pub restGyrLpA: Vec2,
	pub restAccLpB: Vec3,
	pub restAccLpA: Vec2,
	pub kMagRef: f32,
	pub magNormDipLpB: Vec3,
	pub magNormDipLpA: Vec2,
}

impl Default for VQFCoefficients {
	fn default() -> Self {
		Self {
			gyrTs: 0.0,
			accTs: 0.0,
			magTs: 0.0,
			accLpB: Vec3::repeat(f32::NAN),
			accLpA: Vec2::repeat(f32::NAN),
			kMag: -1.0,
			biasP0: -1.0,
			biasV: -1.0,
			biasMotionW: -1.0,
			biasVerticalW: -1.0,
			biasRestW: -1.0,
			restGyrLpB: Vec3::repeat(f32::NAN),
			restGyrLpA: Vec2::repeat(f32::NAN),
			restAccLpB: Vec3::repeat(f32::NAN),
			restAccLpA: Vec2::repeat(f32::NAN),
			kMagRef: -1.0,
			magNormDipLpB: Vec3::repeat(f32::NAN),
			magNormDipLpA: Vec2::repeat(f32::NAN),
		}
	}
}

pub struct Vqf {
	_params: VqfParameters,
	_state: VqfState,
	_coeffs: VQFCoefficients,
}

impl Vqf {
	pub fn new(gyrTs: f32, accTs: Option<f32>, magTs: Option<f32>, params: VqfParameters) -> Vqf {
		let mut vqf = Vqf {
			_params: params,
			_state: Default::default(),
			_coeffs: VQFCoefficients {
				gyrTs,
				accTs: accTs.unwrap_or(gyrTs),
				magTs: magTs.unwrap_or(gyrTs),
				..Default::default()
			},
		};
		vqf._setup();
		vqf
	}

	pub fn updateGyr(&mut self, gyr: Vec3) {
		if self._params.restBiasEstEnabled || self._params.magDistRejectionEnabled {
			let gyrLp = filterVec(
				gyr,
				self._params.restFilterTau,
				self._coeffs.gyrTs,
				self._coeffs.restGyrLpB,
				self._coeffs.restGyrLpA,
				&mut self._state.restGyrLpState,
			);

			let deviation = gyr - gyrLp;
			let squaredDeviation = deviation.dot(&deviation);

			let biasClip = self._params.biasClip * PI / 180.0;
			if squaredDeviation >= (self._params.restThGyr * PI / 180.0).powf(2.0)
				|| gyrLp.abs().max() > biasClip
			{
				self._state.restT = 0.0;
				self._state.restDetected = false;
			}
			self._state.restLastGyrLp = gyrLp;
			self._state.restLastSquaredDeviations[0] = squaredDeviation;
		}

		// remove estimated gyro bias
		let gyrNoBias = gyr - self._state.bias;

		// gyroscope prediction step
		let gyrNorm = gyrNoBias.dot(&gyrNoBias).sqrt();
		let angle = gyrNorm * self._coeffs.gyrTs;
		if gyrNorm > EPS {
			let c = (angle / 2.0).cos();
			let s = (angle / 2.0).sin() / gyrNorm;
			let gyrStepQuat = Quat::from_quaternion(
				[c, s * gyrNoBias[0], s * gyrNoBias[1], s * gyrNoBias[2]].into(),
			);
			self._state.gyrQuat = self._state.gyrQuat * gyrStepQuat;
		}
	}

	pub fn updateAcc(&mut self, acc: Vec3) {
		if acc == Vec3::zeros() {
			return;
		}

		let accTs = self._coeffs.accTs;

		// Rest detection
		if self._params.restBiasEstEnabled {
			let accLp = filterVec(
				acc,
				self._params.restFilterTau,
				accTs,
				self._coeffs.restAccLpB,
				self._coeffs.restAccLpA,
				&mut self._state.restAccLpState,
			);

			let deviation = acc - accLp;
			let squaredDeviation = deviation.dot(&deviation);

			if squaredDeviation >= self._params.restThAcc.powf(2.0) {
				self._state.restT = 0.0;
				self._state.restDetected = false;
			} else {
				self._state.restT += accTs;
				if self._state.restT >= self._params.restMinT {
					self._state.restDetected = true
				}
			}

			self._state.restLastAccLp = accLp;
			self._state.restLastSquaredDeviations[1] = squaredDeviation
		}

		// filter acc in inertial frame
		let accEarth = self._state.gyrQuat * acc;
		self._state.lastAccLp = filterVec(
			accEarth,
			self._params.tauAcc,
			accTs,
			self._coeffs.accLpB,
			self._coeffs.accLpA,
			&mut self._state.accLpState,
		);

		// transform to 6D earth frame and normalize
		let accEarth = (self._state.accQuat * self._state.lastAccLp).normalize();

		// inclination correction
		let q_w = ((accEarth[2] + 1.0) / 2.0).sqrt();
		let accCorrQuat;
		if q_w > EPS {
			accCorrQuat = Quat::from_quaternion(
				[q_w, 0.5 * accEarth[1] / q_w, -0.5 * accEarth[0] / q_w, 0.0].into(),
			);
		} else {
			accCorrQuat = Quat::from_quaternion([0.0, 1.0, 0.0, 0.0].into());
		}
		self._state.accQuat = accCorrQuat * self._state.accQuat;

		// calculate correction angular rate to facilitate debugging
		// self._state.lastAccCorrAngularRate = (accEarth[2]).acos() / self._coeffs.accTs;

		// bias estimation
		if self._params.motionBiasEstEnabled || self._params.restBiasEstEnabled {
			let biasClip = self._params.biasClip * PI / 180.0;
			let mut bias = self._state.bias;

			// get rotation matrix corresponding to accGyrQuat
			let accGyrQuat = self.getQuat6D();
			// ඞ
			let R = accGyrQuat.to_rotation_matrix().into_inner();

			// calculate R*b_hat (only the x and y component, as z is not needed)
			let biasLp = (R * bias).xy();

			// low-pass filter R and R*b_hat
			let mut R = filterVec(
				R.reshape_generic(nalgebra::Const::<9>, nalgebra::Const::<1>),
				self._params.tauAcc,
				accTs,
				self._coeffs.accLpB,
				self._coeffs.accLpA,
				&mut self._state.motionBiasEstRLpState,
			)
			.reshape_generic(nalgebra::Const::<3>, nalgebra::Const::<3>);
			let biasLp = filterVec(
				biasLp,
				self._params.tauAcc,
				accTs,
				self._coeffs.accLpB,
				self._coeffs.accLpA,
				&mut self._state.motionBiasEstBiasLpState,
			);

			// set measurement error and covariance for the respective Kalman filter update
			let e;
			let w;
			if self._state.restDetected && self._params.restBiasEstEnabled {
				e = Some(self._state.restLastGyrLp - bias);
				R = Mat3x3::identity();
				w = Some(Vec3::repeat(self._coeffs.biasRestW));
			} else if self._params.motionBiasEstEnabled {
				// ඞ
				e = Some(Vec3::new(
					-accEarth[1] / accTs + biasLp[0]
						- R[0] * bias[0] - R[1] * bias[1]
						- R[2] * bias[2],
					accEarth[0] / accTs + biasLp[1]
						- R[3] * bias[0] - R[4] * bias[1]
						- R[5] * bias[2],
					-R[6] * bias[0] - R[7] * bias[1] - R[8] * bias[2],
				));
				R = Mat3x3::repeat(f32::NAN);
				w = Some(Vec3::new(
					self._coeffs.biasMotionW,
					self._coeffs.biasMotionW,
					self._coeffs.biasVerticalW,
				));
			} else {
				w = None;
				e = None;
			}

			// Kalman filter update
			// step 1: P = P + V (also increase covariance if there is no measurement update!)
			if self._state.biasP[(0, 0)] < self._coeffs.biasP0 {
				self._state.biasP[(0, 0)] += self._coeffs.biasV;
			}
			if self._state.biasP[(1, 1)] < self._coeffs.biasP0 {
				self._state.biasP[(1, 1)] += self._coeffs.biasV;
			}
			if self._state.biasP[(2, 2)] < self._coeffs.biasP0 {
				self._state.biasP[(2, 2)] += self._coeffs.biasV;
			}

			if let Some(w) = w {
				// clip disagreement to -2..2 °/s
				// (this also effectively limits the harm done by the first inclination correction step)
				let e = e.unwrap();
				let e = Vec3::from_fn(|x, y| e[(x, y)].clamp(-biasClip, biasClip));

				// step 2: K = P R^T inv(W + R P R^T)
				let K = self._state.biasP
					* R.transpose() * (Mat3x3::from_diagonal(&w)
					+ R * self._state.biasP * R.transpose())
				.pseudo_inverse(EPS)
				.unwrap();

				// step 3: bias = bias + K (y - R bias) = bias + K e
				bias += K * e;

				// step 4: P = P - K R P
				self._state.biasP -= K * R * self._state.biasP;

				// clip bias estimate to -2..2 °/s
				bias = Vec3::from_fn(|x, y| bias[(x, y)].clamp(-biasClip, biasClip));
			}

			// ඞ
			self._state.bias = bias;
		}
	}

	pub fn updateMag(&mut self, mag: Vec3) {
		if mag == Vec3::zeros() {
			return;
		}

		let magTs = self._coeffs.magTs;

		// bring magnetometer measurement into 6D earth frame
		let magEarth = self.getQuat6D() * mag;

		if self._params.magDistRejectionEnabled {
			let mut magNormDip = self._state.magNormDip;
			magNormDip[0] = (magEarth.dot(&magEarth)).sqrt();
			magNormDip[1] = -((magEarth[2] / magNormDip[0]).asin());

			if self._params.magCurrentTau > 0.0 {
				magNormDip = filterVec(
					magNormDip,
					self._params.magCurrentTau,
					magTs,
					self._coeffs.magNormDipLpB,
					self._coeffs.magNormDipLpA,
					&mut self._state.magNormDipLpState,
				);
			}

			// magnetic disturbance detection
			if (magNormDip[0] - self._state.magRefNorm).abs()
				< self._params.magNormTh * self._state.magRefNorm
				&& (magNormDip[1] - self._state.magRefDip).abs()
					< self._params.magDipTh * PI / 180.0
			{
				self._state.magUndisturbedT += magTs;

				if self._state.magUndisturbedT >= self._params.magMinUndisturbedTime {
					self._state.magDistDetected = false;
					self._state.magRefNorm +=
						self._coeffs.kMagRef * (magNormDip[0] - self._state.magRefNorm);
					self._state.magRefDip +=
						self._coeffs.kMagRef * (magNormDip[1] - self._state.magRefDip);
				}
			} else {
				self._state.magUndisturbedT = 0.0;
				self._state.magDistDetected = true;
			}

			// new magnetic field acceptance
			if (magNormDip[0] - self._state.magCandidateNorm).abs()
				< self._params.magNormTh * self._state.magCandidateNorm
				&& (magNormDip[1] - self._state.magCandidateDip).abs()
					< self._params.magDipTh * PI / 180.0
			{
				let gyrNorm =
					(self._state.restLastGyrLp.dot(&self._state.restLastGyrLp)).sqrt();
				if gyrNorm >= self._params.magNewMinGyr * PI / 180.0 {
					self._state.magCandidateT += magTs;
				}

				self._state.magCandidateNorm += self._coeffs.kMagRef
					* (magNormDip[0] - self._state.magCandidateNorm);
				self._state.magCandidateDip += self._coeffs.kMagRef
					* (magNormDip[1] - self._state.magCandidateDip);

				if self._state.magDistDetected
					&& (self._state.magCandidateT >= self._params.magNewTime
						|| (self._state.magRefNorm == 0.0
							&& self._state.magCandidateT
								>= self._params.magNewFirstTime))
				{
					self._state.magRefNorm = self._state.magCandidateNorm;
					self._state.magRefDip = self._state.magCandidateDip;
					self._state.magDistDetected = false;
					self._state.magUndisturbedT = self._params.magMinUndisturbedTime;
				}
			} else {
				self._state.magCandidateT = 0.0;
				self._state.magCandidateNorm = magNormDip[0];
				self._state.magCandidateDip = magNormDip[1];
			}
		}

		// calculate disagreement angle based on current magnetometer measurement
		self._state.lastMagDisAngle =
			magEarth[0].atan2(magEarth[1]) - self._state.delta;

		// make sure the disagreement angle is in the range [-pi, pi]
		if self._state.lastMagDisAngle > PI {
			self._state.lastMagDisAngle -= 2.0 * PI;
		} else if self._state.lastMagDisAngle < -PI {
			self._state.lastMagDisAngle += 2.0 * PI;
		}

		let mut k = self._coeffs.kMag;

		if self._params.magDistRejectionEnabled {
			// magnetic disturbance rejection
			if self._state.magDistDetected {
				if self._state.magRejectT <= self._params.magMaxRejectionTime {
					self._state.magRejectT += magTs;
					k = 0.0;
				} else {
					k /= self._params.magRejectionFactor;
				}
			} else {
				self._state.magRejectT = (self._state.magRejectT
					- self._params.magRejectionFactor * magTs)
					.max(0.0);
			}
		}

		// ensure fast initial convergence
		if self._state.kMagInit != 0.0 {
			// make sure that the gain k is at least 1/N, N=1,2,3,... in the first few samples
			if k < self._state.kMagInit {
				k = self._state.kMagInit;
			}

			// iterative expression to calculate 1/N
			self._state.kMagInit = self._state.kMagInit / (self._state.kMagInit + 1.0);

			// disable if t > tauMag
			if self._state.kMagInit * self._params.tauMag < self._coeffs.magTs {
				self._state.kMagInit = 0.0;
			}
		}

		// first-order filter step
		self._state.delta += k * self._state.lastMagDisAngle;
		// calculate correction angular rate to facilitate debugging
		self._state.lastMagCorrAngularRate =
			k * self._state.lastMagDisAngle / self._coeffs.magTs;

		// make sure delta is in the range [-pi, pi]
		if self._state.delta > PI {
			self._state.delta -= 2.0 * PI;
		} else if self._state.delta < -PI {
			self._state.delta += 2.0 * PI;
		}
	}

	pub fn update(&mut self, gyr: Vec3, acc: Vec3, mag: Option<Vec3>) {
		self.updateGyr(gyr);
		self.updateAcc(acc);
		if let Some(mag) = mag {
			self.updateMag(mag);
		}
	}

	pub fn getQuat3D(&self) -> Quat {
		self._state.gyrQuat
	}

	pub fn getQuat6D(&self) -> Quat {
		self._state.accQuat * self.getQuat3D()
	}

	pub fn resetState(&mut self) {
		self._state = Default::default();
		self._state.biasP = self._coeffs.biasP0 * Mat3x3::identity();
		self._state.magRejectT = self._params.magMaxRejectionTime;
	}

	fn _setup(&mut self) {
		let coeffs = &mut self._coeffs;
		let params = &mut self._params;

		assert!(coeffs.gyrTs > 0.);
		assert!(coeffs.accTs > 0.);
		assert!(coeffs.magTs > 0.);

		(coeffs.accLpB, coeffs.accLpA) = filterCoeffs(params.tauAcc, coeffs.accTs);

		coeffs.kMag = gainFromTau(params.tauMag, coeffs.magTs);

		coeffs.biasP0 = (params.biasSigmaInit*100.0).powf(2.);
		// # the system noise increases the variance from 0 to (0.1 °/s)^2 in biasForgettingTime seconds
		coeffs.biasV = (0.1*100.0).powf(2.) * coeffs.accTs/params.biasForgettingTime;
		let pMotion = (params.biasSigmaMotion*100.0).powf(2.);
		coeffs.biasMotionW = pMotion.powf(2.) / coeffs.biasV + pMotion;
		// coeffs.biasVerticalW = coeffs.biasMotionW / max(params.biasVerticalForgettingFactor, 1e-10);
		coeffs.biasVerticalW = coeffs.biasMotionW / params.biasVerticalForgettingFactor.max(1.0e-10);

		let pRest = (params.biasSigmaRest*100.0).powf(2.);
		coeffs.biasRestW = pRest.powf(2.) / coeffs.biasV + pRest;

		(coeffs.restGyrLpB, coeffs.restGyrLpA) = filterCoeffs(params.restFilterTau, coeffs.gyrTs);
		(coeffs.restAccLpB, coeffs.restAccLpA) = filterCoeffs(params.restFilterTau, coeffs.accTs);

		coeffs.kMagRef = gainFromTau(params.magRefTau, coeffs.magTs);
		if params.magCurrentTau > 0. {
				(coeffs.magNormDipLpB, coeffs.magNormDipLpA) = filterCoeffs(params.magCurrentTau, coeffs.magTs);
		}

		self.resetState();
	}
}

// fn quatRotate(q: Quat, v: Vec3) -> Vec3 {
// 	let (q0, q1, q2, q3) = (q.w, q.i, q.j, q.k);
// 	let (v0, v1, v2) = (v.x, v.y, v.z);
// 	let x = (1. - 2.*q2*q2 - 2.*q3*q3)*v0 + 2.*v1*(q2*q1 - q0*q3) + 2.*v2*(q0*q2 + q3*q1);
// 	let y = 2.*v0*(q0*q3 + q2*q1) + v1*(1. - 2.*q1*q1 - 2.*q3*q3) + 2.*v2*(q2*q3 - q1*q0);
// 	let z = 2.*v0*(q3*q1 - q0*q2) + 2.*v1*(q0*q1 + q3*q2) + v2*(1. - 2.*q1*q1 - 2.*q2*q2);
// 	Vec3::new(x, y, z)
// }

fn gainFromTau(tau: f32, Ts: f32) -> f32 {
		assert!(Ts > 0.);
		if tau < 0. {
				0.0  // k=0 for negative tau (disable update)
		} else if tau == 0.0 {
				1.0  // k=1 for tau=0
		} else {
				1.0 - (-Ts/tau).exp()  // fc = 1/(2*pi*tau)
		}
}

fn filterCoeffs(tau: f32, Ts: f32) -> (Vec3, Vec2) {
	assert!(tau > 0.);
	assert!(Ts > 0.);
	// # second order Butterworth filter based on https://stackoverflow.com/a/52764064
	// fc = math.sqrt(2) / (2.0 * math.pi * tau)  # time constant of dampened, non-oscillating part of step response
	let fc = 2.0.sqrt() / (2.0 * PI * tau);

	// C = math.tan(math.pi*fc*Ts)
	let C = (PI * fc * Ts).tan();

	// D = C**2 + math.sqrt(2)*C + 1
	let D = C.powf(2.) + 2.0.sqrt()*C + 1.;
	// b0 = C*C/D
	let b0 = C*C/D;
	// b1 = 2*b0
	let b1 = 2. * b0;
	// b2 = b0
	let b2 = b0;
	// # a0 = 1.0
	// a1 = 2*(C**2-1)/D
	let a1 = 2. * (C.powf(2.) - 1.) / D;
	// a2 = (1-math.sqrt(2)*C+C**2)/D
	let a2 = (1. - 2.0.sqrt() * C + C.powf(2.))/D;
	// return np.array([b0, b1, b2], float), np.array([a1, a2], float)
	(Vec3::new(b0, b1, b2), Vec2::new(a1, a2))
}

fn filterVec<const N: usize, const M: usize>(
	x: nalgebra::SVector<f32, N>,
	tau: f32,
	Ts: f32,
	b: Vec3,
	a: Vec2,
	state: &mut nalgebra::Matrix<
		f32,
		nalgebra::Const<M>,
		nalgebra::Const<N>,
		ArrayStorage<f32, M, N>,
	>,
) -> nalgebra::Matrix<
	f32,
	nalgebra::Const<N>,
	nalgebra::Const<1>,
	ArrayStorage<f32, N, 1>,
> {
	assert!(N >= 2);
	// to avoid depending on a single sample, average the first samples (for duration tau)
	// and then use this average to calculate the filter initial state
	if state[(0, 0)].is_nan() {
		// initialization phase
		if state[(0, 1)].is_nan() {
			// first sample
			state[(0, 1)] = 0.0; // state[0, 1] is used to store the sample count
					 // ඞ
			state.get_mut((1, ..)).unwrap().fill(0.0); // state[1, :] is used to store the sum
		}

		state[(0, 1)] += 1.0;
		let mut out = nalgebra::Matrix::zeros();
		// ඞ
		for (i, x) in x.iter().enumerate() {
			state[(1, i)] += *x;
			out[i] = state[(1, i)] / state[(0, 1)];
		}

		if state[(0, 1)] * Ts >= tau {
			for i in 0..N {
				let init = filterInitialState(out[i], b, a);
				// ඞ
				for j in 0..M {
					state[(j, i)] = init[j];
				}
			}
		}
		return out;
	}

	filterStep(x, b, a, state)
}

fn filterInitialState(x0: f32, b: Vec3, a: Vec2) -> Vec2 {
	Vec2::new(x0 * (1.0 - b[0]), x0 * (b[2] - a[1]))
}

fn filterStep<const N: usize, const M: usize>(
	x: nalgebra::SVector<f32, N>,
	b: Vec3,
	a: Vec2,
	state: &mut nalgebra::Matrix<
		f32,
		nalgebra::Const<M>,
		nalgebra::Const<N>,
		ArrayStorage<f32, M, N>,
	>,
) -> nalgebra::Matrix<
	f32,
	nalgebra::Const<N>,
	nalgebra::Const<1>,
	ArrayStorage<f32, N, 1>,
> {
	// difference equations based on scipy.signal.lfilter documentation
	// assumes that a0 == 1.0
	let y = b[0] * x + nalgebra::Matrix::repeat(state[0]);
	// ඞ
	for i in 0..N {
		state[(0, i)] = b[1] * x[i] - a[0] * y[i] + state[(1, i)];
	}
	// ඞ
	for i in 0..N {
		state[(1, i)] = b[2] * x[i] - a[1] * y[i];
	}
	return y;
}