
#ifndef FISHERKPP_H
#define FISHERKPP_H

#include "hiflow.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

#include "graf/include/graf.h"

// All names are imported for simplicity.
using namespace hiflow;
using namespace hiflow::doffem;
using namespace hiflow::la;
using namespace hiflow::mesh;

// Shorten some datatypes with typedefs.
typedef LADescriptorCoupledD LAD;
typedef LAD::DataType DataType;
typedef LAD::VectorType VectorType;
typedef LAD::MatrixType MatrixType;

// DIM of the problem.
const int DIM = 2;

typedef Vec<DIM, DataType> Coord;

// Rank of the master process.
const int MASTER_RANK = 0;

#include "my_initial_condition.h"

int SCALE_ = 1000;
DataType MAGNITUDE_ = 1e-4;

// GaussianRandomField grf;
// void create_random_field(int rank_){
//   // grf(1500,12);
//   if (rank_ == MASTER_RANK){
//     grf.reinitilize(SCALE_, 11, 100, 1000);
//     LOG_INFO("Compute", "Generating Random Field");
//     // grf.generate_grid(false);
//     grf.generate_grid();
//     //
//     grf.compute("/home/purusharth/Documents/MA-RF-TEMP/ma_hiflow-rf/build/exercises/fischerKPP/gfield.npy");
//     grf.compute();
//   }
//    LOG_INFO("Random Field Max", grf.get_max());
//    LOG_INFO("Random Field Min", grf.get_min());
// }

// Functor used to impose u(x) = c on the boundary.
struct DirichletBC {
  DirichletBC(int dir_mat_number) : dir_mat_num_(dir_mat_number) {}

  void evaluate(const mesh::Entity &face, const Vec<DIM, DataType> &pt_coord,
                std::vector<DataType> &vals) const {
    const int material_number = face.get_material_number();
    vals.clear();

    vals = std::vector<DataType>(0, 0);
    vals.resize(DIM, 0.);
  }

  size_t nb_comp() const { return DIM; }

  size_t nb_func() const { return 1; }

  size_t iv2ind(size_t j, size_t v) const { return v; }

  int dir_mat_num_;
};

// Functor used for the local assembly of the stiffness matrix and load vector.
class LocalPoissonAssembler : private AssemblyAssistant<DIM, DataType> {
public:
  LocalPoissonAssembler(std::vector<std::vector<double>> &vec) : vect(vec) {
    // LOG_INFO("SIZE OF VECTOR", vect.size());
  }
  //   LOG_INFO("Compute", "Generating Random Field");
  //   grf.generate_grid();
  //   grf.compute();
  // }

  void set_parameters(DataType kappa, DataType neumann_bc_val, int dir_mat_num,
                      DataType diffusion_1, DataType carrying_capacity_1,
                      DataType diffusion_2, DataType carrying_capacity_2,
                      DataType ts_, DataType a, DataType b, DataType theta,
                      DataType dt) {
    // std::vector<std::vector<double>>& vec){
    // GaussianRandomField &grf) {
    this->kappa_ = kappa;
    this->neumann_bc_val_ = neumann_bc_val;
    this->dir_mat_num_ = dir_mat_num;
    this->diffusion_1_ = diffusion_1;
    this->carrying_capacity_1_ = carrying_capacity_1;
    this->diffusion_2_ = diffusion_2;
    this->carrying_capacity_2_ = carrying_capacity_2;
    this->time_ = ts_;
    this->a_ = a;
    this->b_ = b;
    this->theta_ = theta;
    this->dt_ = dt;
    // this->vec = vec;
    // this->grf = grf;
  }

  void set_newton_solution(VectorType const *newton_sol) {
    prev_newton_sol_ = newton_sol;
  }

  void set_prev_time_solution(VectorType const *prev_time_sol) {
    prev_time_sol_ = prev_time_sol;
  }

  void operator()(const Element<DataType, DIM> &element,
                  const Quadrature<DataType> &quadrature, LocalMatrix &lm) {
    DataType count = 0;
    const bool need_basis_hessians = false;
    // element.boundary_facet_numbers
    // AssemblyAssistant sets up the local FE basis functions for the current
    // cell
    AssemblyAssistant<DIM, DataType>::initialize_for_element(
        element, quadrature, need_basis_hessians);
    // compute solution values of previous newton iterate u_k at each quadrature
    // point xq: sol_ns_[q] = u_k (xq)
    sol_ns_.clear();
    grad_sol_ns_.clear();
    this->evaluate_fe_function(*prev_newton_sol_, 0, sol_ns_);
    this->evaluate_fe_function_gradients(*prev_newton_sol_, 0, grad_sol_ns_);
    sol_ts_.clear();
    grad_sol_ts_.clear();
    this->evaluate_fe_function(*prev_time_sol_, 0, sol_ts_);
    this->evaluate_fe_function_gradients(*prev_time_sol_, 0, grad_sol_ts_);
    // initilize the local variables
    DataType carrying_capacity = 0.0;
    double diffusion_coeff = 0.0;

    // number of degrees of freedom on current cell
    const size_t num_dof = this->num_dofs_total();
    // number of quadrature points
    const int num_q = this->num_quadrature_points();
    DataType mat_num_ = element.get_cell().get_material_number();

    for (int q = 0; q < num_q; ++q) { // loop over quadrature points

      const DataType wq = w(q); // quadrature weight
      // double D = grf.get_rf_value_int_index((x(q)[0]*SCALE_),
      // (x(q)[1]*SCALE_), 1);
      double D = vect[std::round((x(q)[0] * SCALE_)) - 1]
                     [std::round(x(q)[1] * SCALE_) - 1];
      // std::cout << vect.size() << ": " << D << "\n";
      // std::cout << D << ", ";
      D = D * MAGNITUDE_;
      // std::cout << D << "\n " ;
      // volume element of cell transformation
      const DataType dJ = std::abs(this->detJ(q));

      for (int i = 0; i < num_dof;
           ++i) { // loop over test DOFs <-> test function v
        for (int j = 0; j < num_dof;
             ++j) { // loop over trial DOFs <-> trial function u
          // std::cout << dt_ << " ";
          if (mat_num_ == 2) {
            carrying_capacity = carrying_capacity_1_;
            diffusion_coeff = diffusion_1_;
          }
          if (mat_num_ == 3) {
            // DataType d2 = 0.0;
            // if (time_ > 0.25){
            // d2 = diffusion_2_;
            // std::cout << D << std::endl;
            // }else{
            // d2 = diffusion_2_ * MAGNITUDE_;
            // }
            carrying_capacity = carrying_capacity_2_;
            diffusion_coeff = D;
          }
          lm(i, j) +=
              wq *
              (this->Phi(j, q, 0) * this->Phi(i, q, 0) -
               theta_ * dt_ *
                   (-1.0 * diffusion_coeff *
                        dot(this->grad_Phi(j, q, 0), this->grad_Phi(i, q, 0)) -
                    carrying_capacity * 2 * this->Phi(j, q, 0) * sol_ns_[q] *
                        this->Phi(i, q, 0) +
                    carrying_capacity * this->Phi(j, q, 0) *
                        this->Phi(i, q, 0))) *
              dJ;
        }
      }
    }
  }

  void operator()(const Element<DataType, DIM> &element,
                  const Quadrature<DataType> &quadrature, LocalVector &lv) {
    const bool need_basis_hessians = false;

    // AssemblyAssistant sets up the local FE basis functions for the current
    // cell
    AssemblyAssistant<DIM, DataType>::initialize_for_element(
        element, quadrature,
        need_basis_hessians); // compute solution values of previous newton
                              // iterate u_k at each quadrature
    // point xq: sol_ns_[q] = u_k (xq)
    sol_ns_.clear();
    grad_sol_ns_.clear();
    this->evaluate_fe_function(*prev_newton_sol_, 0, sol_ns_);
    this->evaluate_fe_function_gradients(*prev_newton_sol_, 0, grad_sol_ns_);

    sol_ts_.clear();
    grad_sol_ts_.clear();
    this->evaluate_fe_function(*prev_time_sol_, 0, sol_ts_);
    this->evaluate_fe_function_gradients(*prev_time_sol_, 0, grad_sol_ts_);
    // initilize the local variables
    DataType carrying_capacity = 0.0047;
    double diffusion_coeff = 1.7;

    // number of degrees of freedom on current cell
    const size_t num_dof = this->num_dofs_total();
    const int num_q =
        this->num_quadrature_points(); // number of quadrature points
    DataType mat_num_ = element.get_cell().get_material_number();

    // loop over quadrature points
    for (int q = 0; q < num_q; ++q) {
      const DataType wq = w(q); // quadrature weight
      // double D = grf.get_rf_value_int_index(x(q)[0]*SCALE_, x(q)[1]*SCALE_,
      // 1);
      int xx = std::round(x(q)[0] * SCALE_); //- 1;
      int yy = std::round(x(q)[1] * SCALE_);// - 1;
      // std::cout << xx << ", " << yy << ": ";
      // if (xx < 0 || yy < 0) {
      //   if (xx < 0) {
      //     xx = 0;
      //   }
      //   if (yy < 0) {
      //     yy = 0;
      //   }
      // }
      double D = vect[xx][yy];
      D = D * MAGNITUDE_;
      // std::cout << "D =  " << D << std::endl;
      // std::cout << D << " "; ~0.00110755 -> Mean: 100, variance: 50,
      // MAGNITUDE 1e-5:
      const DataType dJ =
          std::abs(this->detJ(q)); // volume element of cell transformation

      for (int i = 0; i < num_dof; ++i) {
        if (mat_num_ == 2) {
          carrying_capacity = carrying_capacity_1_;
          diffusion_coeff = diffusion_1_;
        }
        if (mat_num_ == 3) {
          // DataType d2 = 0.0;
          // if (time_ > 0.25){
          // std::cout << "time inside yaay! ";
          // d2 = diffusion_2_;
          // }else{
          // d2 = diffusion_2_ * MAGNITUDE_;
          // }
          carrying_capacity = carrying_capacity_2_;
          diffusion_coeff = D;
          // std::cout << diffusion_coeff << std::endl;
        }
        // std::cout << "(" << x(q)[0] << ", " << x(q)[1] << ") " <<
        // D << " * " << diffusion_2_ << " = " << diffusion_coeff << std::endl;
        lv[i] += wq *
                 ((sol_ns_[q] - sol_ts_[q]) * this->Phi(i, q, 0) -
                  theta_ * dt_ *
                      (-1.0 * diffusion_coeff *
                           dot(grad_sol_ns_[q], this->grad_Phi(i, q, 0)) -
                       carrying_capacity * sol_ns_[q] * sol_ns_[q] *
                           this->Phi(i, q, 0) +
                       carrying_capacity * sol_ns_[q] * this->Phi(i, q, 0)) -
                  (1 - theta_) * dt_ *
                      (-1.0 * diffusion_coeff *
                           dot(grad_sol_ts_[q], this->grad_Phi(i, q, 0)) -
                       carrying_capacity * sol_ts_[q] * sol_ts_[q] *
                           this->Phi(i, q, 0) +
                       carrying_capacity * sol_ts_[q] * this->Phi(i, q, 0))) *
                 dJ;
      }
    }
  }

  FunctionValues<DataType> sol_ns_; // solution at previous newton step
  FunctionValues<DataType> sol_ts_;
  FunctionValues<Vec<DIM, DataType>>
      grad_sol_ns_; // gradient of solution at previous newton step
  FunctionValues<Vec<DIM, DataType>> grad_sol_ts_;

  DataType kappa_, neumann_bc_val_;
  int dir_mat_num_;
  VectorType const *prev_newton_sol_;
  VectorType const *prev_time_sol_;
  DataType diffusion_1_;
  DataType carrying_capacity_1_;
  DataType diffusion_2_;
  DataType carrying_capacity_2_;
  DataType time_;
  DataType a_, b_;
  DataType theta_;
  DataType dt_;
  std::vector<std::vector<double>> &vect;
  // GaussianRandomField grf;

}; // end of assembler

#endif
