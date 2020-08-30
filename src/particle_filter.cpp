/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  if (is_initialized) {
     return;
  }
  
  num_particles = 100;  // TODO: Set the number of particles
  particles.resize(num_particles);
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std[0]);
  std::normal_distribution<double> dist_y(0, std[1]);
  std::normal_distribution<double> dist_theta(0, std[2]);
  
  for(int i = 0; i < num_particles; i++) {
     particles[i].id = i;
     particles[i].x = x + dist_x(gen);
     particles[i].y = y + dist_y(gen);
     particles[i].theta = theta + dist_theta(gen);
     particles[i].weight = 1;
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(0, std_pos[0]);
    std::normal_distribution<double> dist_y(0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0, std_pos[2]);
    
    for(int i = 0; i < num_particles; i++) {
        
        if ( fabs(yaw_rate) < 0.00001) { // When yaw doesn't change
          particles[i].x += velocity * delta_t * cos( particles[i].theta );
          particles[i].y += velocity * delta_t * sin( particles[i].theta );
          
        }
        else {
            particles[i].x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta +
                                              (yaw_rate * delta_t)) - sin(particles[i].theta)) + dist_x(gen);
            particles[i].y = particles[i].y + (velocity/yaw_rate) * (-cos(particles[i].theta +
                                              (yaw_rate * delta_t)) + cos(particles[i].theta)) + dist_y(gen);
            particles[i].theta = particles[i].theta + (yaw_rate * delta_t) +  dist_theta(gen);
        }        
    }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
    int landMarkId;
    for (int i = 0; i < static_cast<int>(observations.size()); i++) {
        double dist = 0;    
        bool firstIteration = true;
        for (int j = 0; j < static_cast<int>(predicted.size()); j++) {
            double newDistance = sqrt(pow((observations[i].x - predicted[j].x),2) + 
                                      pow((observations[i].y - predicted[j].y),2));            
            if (firstIteration) {
                dist = newDistance;
                landMarkId = predicted[j].id;
                firstIteration = false;
            }
            else if (newDistance < dist) {
                dist = newDistance;
                landMarkId = predicted[j].id;
            }
        }        
        observations[i].id = landMarkId;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
    for (int i = 0; i < num_particles; i++) {  
        double particleX = particles[i].x;
        double particleY = particles[i].y;
        double theta = particles[i].theta;
        
        // Look for landmarks in Particle's Range      
        vector<LandmarkObs> landMarksInRange;
        
        // Loop through all the landmarks
        for(int j = 0; j < static_cast<int>(map_landmarks.landmark_list.size()); j++) {
            double landmarkX = map_landmarks.landmark_list[j].x_f;
            double landmarkY = map_landmarks.landmark_list[j].y_f;
            int id = map_landmarks.landmark_list[j].id_i;
            
            // dist between particle and landmark
            double dX = particleX - landmarkX;
            double dY = particleY - landmarkY;
            if ( sqrt(dX*dX + dY*dY) <= sensor_range ) {
                // add landmark to vector
                landMarksInRange.push_back(LandmarkObs{ id, landmarkX, landmarkY });
            }
        }
        
        // Transform observation coordinates.
        vector<LandmarkObs> mappedObservations; // mapped observations wrt to each particle
        for(unsigned int j = 0; j < observations.size(); j++) {
            double xm = cos(theta)*observations[j].x - sin(theta)*observations[j].y + particleX;
            double ym = sin(theta)*observations[j].x + cos(theta)*observations[j].y + particleY;
            mappedObservations.push_back(LandmarkObs{ observations[j].id, xm, ym });
        }
        
        // Associate landMark to observations
        dataAssociation(landMarksInRange, mappedObservations);        
       
        particles[i].weight = 1.0;
        // Calculate weights.
        for(unsigned int j = 0; j < mappedObservations.size(); j++) {
            double observationX = mappedObservations[j].x;
            double observationY = mappedObservations[j].y;
            
            int landmarkId = mappedObservations[j].id;
            
            double landmarkX, landmarkY;
            unsigned int k = 0;           
            bool isLandMarkFound = false;
            
            while( !isLandMarkFound && (k < landMarksInRange.size()) ) {
                if ( landMarksInRange[k].id == landmarkId) {
                    isLandMarkFound = true;
                    landmarkX = landMarksInRange[k].x;
                    landmarkY = landMarksInRange[k].y;
                }
                k++;
            }            
            // Calculating weight.
            double dX = observationX - landmarkX;
            double dY = observationY - landmarkY;
            
            double weight = ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp( -( dX*dX/(2*std_landmark[0]*std_landmark[1]) + 
                                                                              (dY*dY/(2*std_landmark[0]*std_landmark[1])) ) );
            if (weight == 0) {
                particles[i].weight *= 0.000001;
            } else {
                particles[i].weight *= weight;
            }
        }
    }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    // Get weights and max weight.
    vector<double> weights;
    double maxWeight = std::numeric_limits<double>::min();
    for(int i = 0; i < num_particles; i++) {
      weights.push_back(particles[i].weight);
      if ( particles[i].weight > maxWeight ) {
        maxWeight = particles[i].weight;
      }
    }
  
    // Creating distributions.
    std::default_random_engine gen;
    std::uniform_real_distribution<double> distWeights(0.0, maxWeight);
    std::uniform_int_distribution<int> distIndexes(0, num_particles - 1);
  
    // Generating index.
    int index = distIndexes(gen);
  
    double beta = 0.0;
  
    // the wheel
    vector<Particle> resampledParticles(num_particles);
    for(int i = 0; i < num_particles; i++) {
      beta += distWeights(gen) * 2.0;
      while( beta > weights[index]) {
        beta -= weights[index];
        index = (index + 1) % num_particles;
      }
      resampledParticles.push_back(particles[index]);
    }  
    particles = resampledParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}