"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
from numpy.linalg import multi_dot


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0.0, 0.0])  # state

        self.P_t = np.array(
            [[1000, 0, 0, 0], [0, 1000, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1000]]
        )

        # The state transition matrix - the dynamics model
        self.D_t = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])

        # The measurement model. Since its 2x4, it tells us that velocities are not measured.
        self.M_t = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        self.Q = Q
        self.R = R

    def predict(self):
        self.state = np.dot(self.D_t, self.state)

        self.P_t = multi_dot([self.D_t, self.P_t, self.D_t.T]) + self.Q

    def correct(self, meas_x, meas_y):
        # measurement update
        Z = np.array([meas_x, meas_y])
        y = Z.T - self.M_t.dot(self.state)
        S = multi_dot([self.M_t, self.P_t, self.M_t.T]) + self.R
        K = multi_dot([self.P_t, self.M_t.T, np.linalg.inv(S)])

        self.state = self.state + K.dot(y)
        self.P_t = (np.identity(4) - K.dot(self.M_t)).dot(self.P_t)

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get("num_particles")  # required by the autograder
        self.sigma_exp = kwargs.get("sigma_exp")  # required by the autograder
        self.sigma_dyn = kwargs.get("sigma_dyn")  # required by the autograder
        self.template_rect = kwargs.get("template_coords")  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame
        self.particles = self._init_particles()
        self.weights = np.ones(self.num_particles) / self.num_particles

    def _init_particles(self):
        particle_array = np.zeros((self.num_particles, 2))

        y0 = self.template_rect["y"]
        y1 = y0 + self.template_rect["h"]

        x0 = self.template_rect["x"]
        x1 = x0 + self.template_rect["w"]

        particle_array[:, 0] = np.random.randint(x0, x1, size=self.num_particles)
        particle_array[:, 1] = np.random.randint(y0, y1, size=self.num_particles)

        return particle_array.astype(np.uint16)

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        error = template.astype(np.float32) - frame_cutout.astype(np.float32)
        mse = np.square(error).mean()

        error_metric = np.exp(-mse / (2 * self.sigma_exp ** 2))

        return error_metric

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        particles = np.copy(self.particles)

        indexes = np.random.choice(
            self.num_particles, size=self.num_particles, p=self.weights
        )

        return particles[indexes]

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        new_particles = np.copy(self.particles)
        new_weights = np.zeros(self.num_particles)
        norm = 0

        # Get new data
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        for i, _ in enumerate(new_particles):

            # Prediction
            x_d = np.random.normal(scale=self.sigma_dyn)
            y_d = np.random.normal(scale=self.sigma_dyn)

            x, y = new_particles[i]

            new_particles[i, 0] = x + x_d
            new_particles[i, 1] = y + y_d
            ##############################################################################

            # Measurement
            x, y = new_particles[i]

            row_start = y - self.template_rect["h"] // 2
            row_end = row_start + self.template_rect["h"]

            col_start = x - self.template_rect["w"] // 2
            col_end = col_start + self.template_rect["w"]

            frame_cutout = frame_gray[row_start:row_end, col_start:col_end]

            if not frame_cutout.shape == template_gray.shape:
                error_calc = 0
            else:
                error_calc = self.get_error_metric(template_gray, frame_cutout)

            new_weights[i] = error_calc
            norm += error_calc
            ##############################################################################

        self.particles = new_particles
        self.weights = new_weights / norm
        self.best_particle = self.particles[self.weights.argmax()]
        self.particles = self.resample_particles()

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        x_weighted_mean = int(x_weighted_mean)
        y_weighted_mean = int(y_weighted_mean)

        # Complete the rest of the code as instructed.
        self._draw_particle_and_spread(frame_in, x_weighted_mean, y_weighted_mean)

        self._draw_tracking_window(frame_in, x_weighted_mean, y_weighted_mean)

    def _draw_particle_and_spread(self, frame_in, x_weighted_mean, y_weighted_mean):
        distances = np.zeros(self.num_particles)

        # draw particles
        for i, particle in enumerate(self.particles):
            color = (0, 255, 0)
            radius = 1

            cv2.circle(frame_in, tuple(particle[:2]), radius, color, -1)

            x, y = particle[:2]
            dst = np.sqrt((x - x_weighted_mean) ** 2 + (y - y_weighted_mean) ** 2)
            distances[i] = dst

        # std deviation calculations
        radius = np.round(distances.std())
        color = (0, 0, 255)

        cv2.circle(frame_in, (x_weighted_mean, y_weighted_mean), int(radius), color, 1)

    def _draw_tracking_window(self, frame_in, x_weighted_mean, y_weighted_mean):
        # draw template box
        h, w, _ = self.template.shape

        row_start = int(y_weighted_mean - h // 2)
        row_end = row_start + h

        col_start = int(x_weighted_mean) - w // 2
        col_end = col_start + w

        cv2.rectangle(
            frame_in,
            (col_start, row_start),
            (col_end, row_end),
            (255, 0, 0),
            1,
        )


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(
            frame, template, **kwargs
        )  # call base class constructor

        self.alpha = kwargs.get("alpha")  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        super(AppearanceModelPF, self).process(frame)

        x, y = self.best_particle.astype(np.uint16)

        row_start = y - self.template_rect["h"] // 2
        row_end = row_start + self.template_rect["h"]

        col_start = x - self.template_rect["w"] // 2
        col_end = col_start + self.template_rect["w"]

        best = frame[row_start:row_end, col_start:col_end, :]
        # import pdb; pdb.set_trace()
        template_temp = self.alpha * best + (1 - self.alpha) * self.template
        self.template = template_temp.astype(np.uint8)

        # tmp_frame = np.copy(frame)
        # cv2.rectangle(
        #     tmp_frame,
        #     (col_start, row_start),
        #     (col_end, row_end),
        #     (0, 0, 255),
        #     2,
        # )
        #
        # cv2.imshow('update template', tmp_frame)
        # cv2.waitKey(1)


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.particles = np.zeros((self.num_particles, 3), dtype=np.uint16)
        self.particles[:,:2] = self._init_particles()
        self.particles[:, 2] = np.ones(self.num_particles) * 100
        self.min_scale = kwargs.get('min_scale', 1)


    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        # super(MDParticleFilter, self).process(frame)

        new_particles = np.copy(self.particles)
        new_weights = np.zeros(self.num_particles)
        norm = 0

        # Get new data
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        for i, _ in enumerate(new_particles):

            # Prediction
            x_d = np.random.normal(scale=self.sigma_dyn)
            y_d = np.random.normal(scale=self.sigma_dyn)
            z_d = np.random.normal(scale=self.sigma_dyn)

            x, y, z = new_particles[i]

            new_particles[i, 0] = x + x_d
            new_particles[i, 1] = y + y_d
            new_particles[i, 2] = max(z + z_d, self.min_scale*100)

            # Measurement
            x, y, z = new_particles[i]

            # scale template
            scale_factor = z / 100

            template_gray_scaled = cv2.resize(
                template_gray, (0, 0), fx=scale_factor, fy=scale_factor
            )

            h, w = template_gray_scaled.shape

            row_start = y - h // 2
            row_end = row_start + h

            col_start = x - w // 2
            col_end = col_start + w

            frame_cutout = frame_gray[row_start:row_end, col_start:col_end]

            if frame_cutout.shape != template_gray_scaled.shape:
                similarity_score = 0
            else:
                similarity_score = self.get_error_metric(template_gray_scaled, frame_cutout)

            new_weights[i] = similarity_score
            norm += similarity_score

        self.particles = new_particles
        self.weights = new_weights / norm

        self.particles = self.resample_particles()


    def _draw_tracking_window(self, frame_in, x_weighted_mean, y_weighted_mean):

        z_weighted_mean = np.average(self.particles[:, 2], weights=self.weights)/100

        template_mean = cv2.resize(
            self.template[:,:,0], (0, 0), fx=z_weighted_mean, fy=z_weighted_mean
        )
        h, w = template_mean.shape

        row_start = y_weighted_mean - h // 2
        row_end = row_start + h

        col_start = x_weighted_mean - w // 2
        col_end = col_start + w

        cv2.rectangle(
            frame_in,
            (col_start, row_start),
            (col_end, row_end),
            (0, 0, 255),
            1,
        )
