
#include "para_fischerKPP.h"

#ifdef WITH_GPERF
#include "profiler.h"
#endif

static const char *PARAM_FILENAME = "para_fischerKPP.xml";
#ifndef MESHES_DATADIR
#define MESHES_DATADIR "./"
#endif
static const char *DATADIR = MESHES_DATADIR;

//@TODO: Change class name

#include "graf/include/graf.h"

// Main application class ///////////////////////////////////

int SIZE_ = 1001;


// int load_get_std_vector(std::string filename, std::vector<std::vector<double>> &vec);
std::vector<double> flattened_array(SIZE_ *SIZE_);
std::vector<std::vector<double>> vec2;
class FisherKolmogorovGrowth
    : public NonlinearProblem<LAD::MatrixType, LAD::VectorType> {
public:
  FisherKolmogorovGrowth(const std::string &param_filename,
                         const std::string &path_mesh)
      : path_mesh(path_mesh), comm_(MPI_COMM_WORLD), rank_(-1),
        num_partitions_(-1),
        params_(param_filename, MASTER_RANK, MPI_COMM_WORLD), is_done_(false),
        refinement_level_(0) {
    // grf(1500,15) { // random field
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &num_partitions_);

    // Setup Parallel Output / Logging
    if (rank_ == 0) {
      INFO = true;
      // create_random_field(0);
      // this->grf.reinitilize(SIZE_, 11, 100, 1000);
      // LOG_INFO("Compute", "Generating Random Field");
      // // grf.generate_grid(false);
      // this->grf.generate_grid();
      // // grf.compute("/home/purusharth/Documents/MA-RF-TEMP/ma_hiflow-rf/build/exercises/fischerKPP/gfield.npy");
      // this->grf.compute();
      // LOG_INFO("Random Field Max", grf.get_max());
      // LOG_INFO("Random Field Min", grf.get_min());
      // this->grf.get_std_vec(vec);
      // int index = 0;
      // for (int i = 0; i < SIZE_; i++) {
      //   for (int j = 0; j < SIZE_; j++) {
      //     flattened_array[i * SIZE_ + j] = vec[i][j];
      //   }
      // }
      // LOG_INFO("SIZE OF VECTOR", );
      // create random filed and save npy
    }else  {
      // read npy
      //  for (int i = 0; i < SIZE_; i++) {
      //     for (int j = 0; j < SIZE_; j++) {
      //         vec[i][j] = flattened_array[i * SIZE_ + j];
      //  }
      // }
      INFO = false;
    }


  }

  // Main algorithm

  void run() {

    // Construct / read in the initial mesh.

    // this->grf.generate_grid();
    // this->grf.compute();
    build_initial_mesh();
    
    
    // Main adaptation loop.
    ts_ = 0;
    t_ = 0;
    DataType t_end = params_["Timestepping"]["total_time"].get<double>();
    DataType step_size = params_["Timestepping"]["step_size"].get<double>();
    dt_ = step_size;
    LOG_INFO("Total Time", t_end);
    LOG_INFO("Timestep", step_size)
    LOG_INFO("", "");
    LOG_INFO("", "");
    LOG_INFO("", "===========================");
    LOG_INFO("mesh level", refinement_level_);
    LOG_INFO("", "===========================");
    // MPI_Bcast(flattened_array.data(), SIZE_ * SIZE_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // if (rank_ != 0) {
    //   std::cout << "Inside a non root processe" << std::endl;
    //   std::cout << "FLAT ARRAY: " << flattened_array[254] << std::endl;
    //   vec.resize(SIZE_, std::vector<double>(SIZE_));
    //   for (int i = 0; i < SIZE_; i++) {
    //     for (int j = 0; j < SIZE_; j++) {
    //       vec[i][j] = flattened_array[i * SIZE_ + j];
    //     }
    //   }
    // }

    Timer timer;
    timer.start();
    // Initialize space and linear algebra.
    LOG_INFO("do", "Prepare System");
    prepare_system();
    assembly_mass_matrix();
    initial_condition();
    visualize();
    timer.stop();
    LOG_INFO("duration", timer.get_duration());

    timer.reset();
    timer.start();
    while (t_ < t_end) {
      ts_ = ts_ + 1;
      t_ = t_ + step_size;
      LOG_INFO("Time", t_);
      LOG_INFO("Time Step ", ts_);
      LOG_INFO("do", "Solve System "); // Solve the linear system.
      solve_system();
      timer.stop();
      LOG_INFO("duration", timer.get_duration());

      if (ts_ % visualization_frequency_ == 0.) {
        timer.reset();
        timer.start();
        LOG_INFO(
            "do",
            "Visualize Solution "); // Visualize the solution and the errors.
        visualize();
        timer.stop();
        LOG_INFO("duration", timer.get_duration());
      } else {
        LOG_INFO("Skipping Visualization for timestep ", ts_);
      }
    }
  }

  ~FisherKolmogorovGrowth() {}

private:
  // -----------------  Member functions
  std::string path_mesh;
  void assembly_mass_matrix();
  void initial_condition();
  void build_initial_mesh(); // Read and distribute mesh.
  void prepare_system(); // Setup space, linear algebra, and compute Dirichlet
                         // values.
  // assembler routines for Jacobian and residual in Newton's method
  void EvalGrad(const VectorType &in, MatrixType *out);
  void EvalFunc(const VectorType &in, VectorType *out);
  void solve_system(); // Compute solution x.
  void visualize();    // Visualize the results.
  void adapt();        // Adapt the space (mesh and/or degree).

  // -----------------  Member variables
  MPI_Comm comm_;             // MPI communicator.
  int rank_, num_partitions_; // Local process rank and number of processes.
  int visualization_frequency_;
  int ts_; // timestep
  double dt_, t_;
  bool is_done_;         // Current refinement level.
  int refinement_level_; // Dof id:s for Dirichlet boundary conditions.
  std::vector<int>
      dirichlet_dofs_; // Dof values for Dirichlet boundary conditions.
  std::vector<DataType> dirichlet_values_;

  PropertyTree params_; // Parameter data read in from file.
  MeshPtr mesh_, mesh_without_ghost_,
      master_mesh_;                  // Local mesh and mesh on master process.
  VectorSpace<DataType, DIM> space_; // Solution space.
  MatrixType matrix_;                // System matrix.
  MatrixType mass_sys_;
  CG<LAD> cg_init_;
  StandardGlobalAssembler<DataType, DIM> global_asm_; // Global assembler.
  Newton<LAD, DIM> newton_;                           // nonlinear solver
  LinearSolver<LAD> *solver_;                         // linear solver
  PreconditionerBlockJacobiExt<LAD> precond_;         // preconditioner
  VectorType rhs_, sol_; // Vectors for solution and load vector.
  VectorType prev_sol_;
  DataType kappa_; // Flag for stopping adaptive loop.
  DataType a_, b_;
  DataType mean_, variance_, normed_;
  DataType theta_;

  GaussianRandomField grf;
}; // end class FisherKolmogorovGrowth

// Program entry point

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  // set default parameter file
  std::string param_filename(PARAM_FILENAME);
  std::string path_mesh;
  // if set take parameter file specified on console
  if (argc > 1) {
    param_filename = std::string(argv[1]);
  }
  // if set take mesh following path specified on console
  if (argc > 2) {
    path_mesh = std::string(argv[2]);
  }
  try {
    LogKeeper::get_log("info").set_target(&(std::cout));
    LogKeeper::get_log("debug").set_target(&(std::cout));
    LogKeeper::get_log("error").set_target(&(std::cout));
    LogKeeper::get_log("warning").set_target(&(std::cout));

    // Create application object and run it
    FisherKolmogorovGrowth app(param_filename, path_mesh);
    app.run();

  } catch (std::exception &e) {
    std::cerr << "\nProgram ended with uncaught exception.\n";
    std::cerr << e.what() << "\n";
    return -1;
  }

#ifdef WITH_GPERF
  ProfilerStop();
#endif
  MPI_Finalize();
  return 0;
}

void FisherKolmogorovGrowth::assembly_mass_matrix() {
  MassMatrixAssemblerRD local_asm;
  global_asm_.assemble_matrix(space_, local_asm, mass_sys_);
}

void FisherKolmogorovGrowth::initial_condition() {
  VectorType rhs2;

  rhs2.Init(comm_, space_.la_couplings());
  rhs2.Zeros();

  InitialConditionAssembler local_asm;
  local_asm.set_parameters(mean_, variance_, normed_);

  global_asm_.assemble_vector(space_, local_asm, rhs2);

  cg_init_.SetupOperator(mass_sys_);

  cg_init_.Solve(rhs2, &prev_sol_);

  prev_sol_.Update();

  if (!dirichlet_dofs_.empty()) {
    // correct solution with dirichlet BC
    prev_sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(),
                        vec2ptr(dirichlet_values_));
  }
  sol_.CloneFrom(prev_sol_);
}

void FisherKolmogorovGrowth::build_initial_mesh() {
  mesh::MeshImpl impl = mesh::MeshImpl::DbView;

  // Read in the mesh on the master process. The mesh is chosen according to the
  // DIM of the problem.
  if (rank_ == MASTER_RANK) {
    std::string mesh_name;

    switch (DIM) {
    case 2: {
      mesh_name =
          params_["Mesh"]["Filename2"].get<std::string>("unit_square.inp");
      break;
    }
    case 3: {
      mesh_name =
          params_["Mesh"]["Filename3"].get<std::string>("unit_cube.inp");
      break;
    }

    default:
      assert(0);
    }
    std::string mesh_filename;
    if (path_mesh.empty()) {
      mesh_filename = std::string(DATADIR) + mesh_name;
    } else {
      mesh_filename = path_mesh + mesh_name;
    }

    std::vector<MasterSlave> period(0, MasterSlave(0., 0., 0., 0));
    // read the mesh
    master_mesh_ =
        read_mesh_from_file(mesh_filename, DIM, DIM, 0, impl, period);

    // Refine the mesh until the initial refinement level is reached.
    const int initial_ref_lvl = params_["Mesh"]["InitialRefLevel"].get<int>(3);

    if (initial_ref_lvl > 0) {
      master_mesh_ = master_mesh_->refine_uniform_seq(initial_ref_lvl);
    }
    refinement_level_ = initial_ref_lvl;
  }

  MPI_Bcast(&refinement_level_, 1, MPI_INT, MASTER_RANK, comm_);

  // partition the mesh and distribute the subdomains across all processes
  mesh_without_ghost_ =
      partition_and_distribute(master_mesh_, MASTER_RANK, comm_);

  assert(mesh_without_ghost_ != 0);

  // compute ghost cells
  SharedVertexTable shared_verts;
  mesh_ = compute_ghost_cells(*mesh_without_ghost_, comm_, shared_verts, impl);
}

void FisherKolmogorovGrowth::prepare_system() {
  // Assign degrees to each element.
  const int nb_fe_var = 1;

  const int fe_degree = params_["FESpace"]["FeDegree"].get<int>(1);
  std::vector<int> fe_params(nb_fe_var, fe_degree);

  std::vector<FEType> fe_ansatz(nb_fe_var, FEType::LAGRANGE);
  std::vector<bool> is_cg(nb_fe_var, true);

  // Initialize the VectorSpace object.
  space_.Init(*mesh_, fe_ansatz, is_cg, fe_params,
              hiflow::doffem::DOF_ORDERING::HIFLOW_CLASSIC);

  LOG_INFO("nb nonlin trafos", space_.fe_manager().nb_nonlinear_trafos());

  // Compute the matrix sparsity structure
  SparsityStructure sparsity;
  compute_sparsity_structure(space_, sparsity, false, false);

  // initialize matrix object
  matrix_.Init(comm_, space_.la_couplings());
  matrix_.InitStructure(sparsity);
  matrix_.Zeros();

  mass_sys_.Init(comm_, space_.la_couplings());
  mass_sys_.InitStructure(sparsity);
  mass_sys_.Zeros();

  cg_init_.InitControl(5000, 1e-8, 1e-6, 1e5);
  cg_init_.SetupOperator(mass_sys_);
  cg_init_.SetPrintLevel(2);

  // initialize vector objects for solution coefficitons and
  // right hand side
  sol_.Init(comm_, space_.la_couplings());
  prev_sol_.Init(comm_, space_.la_couplings());
  rhs_.Init(comm_, space_.la_couplings());
  rhs_.Zeros();
  sol_.Zeros();
  prev_sol_.Zeros();

  // Compute Dirichlet BC dofs and values using known exact solution.
  dirichlet_dofs_.clear();
  dirichlet_values_.clear();

  DirichletBC bc_dirichlet(
      params_["Equation"]["DirichletMaterialNumber"].get<int>());

  compute_dirichlet_dofs_and_values(bc_dirichlet, space_, 0, dirichlet_dofs_,
                                    dirichlet_values_);
  // apply BC to initial solution
  if (!dirichlet_dofs_.empty()) {
    // correct solution with dirichlet BC
    sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(),
                   vec2ptr(dirichlet_values_));
    prev_sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(),
                        vec2ptr(dirichlet_values_));
  }

  // setup  linear solver
  LinearSolverFactory<LAD> SolFact;
  solver_ =
      SolFact.Get(params_["LinearSolver"]["Name"].get<std::string>("GMRES"))
          ->params(params_["LinearSolver"]);
  solver_->SetupOperator(matrix_);

  precond_.Init_ILU_pp();
  solver_->SetupPreconditioner(precond_);

  // get nonlinear solver parameters from param file
  int nls_max_iter = params_["NonlinearSolver"]["MaximumIterations"].get<int>();
  DataType nls_abs_tol =
      params_["NonlinearSolver"]["AbsoluteTolerance"].get<DataType>();
  DataType nls_rel_tol =
      params_["NonlinearSolver"]["RelativeTolerance"].get<DataType>();
  DataType nls_div_tol =
      params_["NonlinearSolver"]["DivergenceLimit"].get<DataType>();
  std::string forcing_strategy =
      params_["NonlinearSolver"]["ForcingStrategy"].get<std::string>();
  bool use_forcing_strategy = (forcing_strategy != "None");
  DataType eta = 1.e-4; // initial value of forcing term

  // get forcing strategy parameters from param file
  DataType eta_initial =
      params_["NonlinearSolver"]["InitialValueForcingTerm"].get<DataType>();
  DataType eta_max =
      params_["NonlinearSolver"]["MaxValueForcingTerm"].get<DataType>();
  DataType gamma_EW2 =
      params_["NonlinearSolver"]["GammaParameterEW2"].get<DataType>();
  DataType alpha_EW2 =
      params_["NonlinearSolver"]["AlphaParameterEW2"].get<DataType>();

  a_ = params_["Equation"]["a"].get<DataType>();
  b_ = params_["Equation"]["b"].get<DataType>();
  visualization_frequency_ =
      params_["Timestepping"]["VisualizationFrequency"].get<int>(1);
  mean_ = params_["InitialCondition"]["Mean"].get<DataType>();
  variance_ = params_["InitialCondition"]["Variance"].get<DataType>();
  normed_ = params_["InitialCondition"]["Normed"].get<bool>(false);
  theta_ = params_["NonlinearSolver"]["Theta"].get<DataType>();

  // setup nonlinear solver
  newton_.InitParameter(&rhs_, &matrix_);
  newton_.InitParameter(Newton<LAD, DIM>::NewtonInitialSolutionOwn);
  newton_.InitControl(nls_max_iter, nls_abs_tol, nls_rel_tol, nls_div_tol);
  newton_.SetOperator(*this);
  newton_.SetLinearSolver(*this->solver_);
  newton_.SetPrintLevel(1);

  // Forcing strategy object
  if (forcing_strategy == "EisenstatWalker1") {
    EWForcing<LAD> *EW_Forcing = new EWForcing<LAD>(eta_initial, eta_max, 1);
    newton_.SetForcingStrategy(*EW_Forcing);
  } else if (forcing_strategy == "EisenstatWalker2") {
    EWForcing<LAD> *EW_Forcing =
        new EWForcing<LAD>(eta_initial, eta_max, 2, gamma_EW2, alpha_EW2);
    newton_.SetForcingStrategy(*EW_Forcing);
  }
}

// Assemble jacobian Matrix for Newton method for solving F(u) = 0 out = D_u
// F[in]
void FisherKolmogorovGrowth::EvalGrad(const VectorType &in, MatrixType *out) {
  // update ghost values of input vector (related to parallelisation)
  // in.Update();

  // pass parameters to local assembler
  LocalPoissonAssembler local_asm(vec2  );
  local_asm.set_parameters(
      this->kappa_, params_["Equation"]["NeumannBC"].get<DataType>(),
      params_["Equation"]["DirichletMaterialNumber"].get<int>(),
      params_["Parameters"]["diffusion_1"].get<DataType>(),
      params_["Parameters"]["carrying_capacity_1"].get<DataType>(),
      params_["Parameters"]["diffusion_2"].get<DataType>(),
      params_["Parameters"]["carrying_capacity_2"].get<DataType>(), t_, a_, b_,
      theta_, dt_);

  local_asm.set_prev_time_solution(&prev_sol_);

  // pass current Newton iterate to local assembler
  local_asm.set_newton_solution(&in);

  // call assemble routine
  this->global_asm_.should_reset_assembly_target(true); // false
  // this->global_asm_.assemble_matrix_boundary(space_, local_asm, matrix_);
  // global_asm_.should_reset_assembly_target(true);

  this->global_asm_.assemble_matrix(this->space_, local_asm, *out);

  // Correct Dirichlet dofs.
  if (!this->dirichlet_dofs_.empty()) {
    out->diagonalize_rows(vec2ptr(this->dirichlet_dofs_),
                          this->dirichlet_dofs_.size(), 1.0);
  }

  // update matrix factorization, used in preconditioner
  this->precond_.SetupOperator(*out);
  this->precond_.Build();
}

void FisherKolmogorovGrowth::EvalFunc(const VectorType &in, VectorType *out) {

  params_["Parameters"]["diffusion_1"].get<DataType>();
  params_["Parameters"]["carrying_capacity_1"].get<DataType>();
  params_["Parameters"]["diffusion_2"].get<DataType>();
  params_["Parameters"]["carrying_capacity_2"].get<DataType>();

  LocalPoissonAssembler local_asm(vec2);
  local_asm.set_parameters(
      this->kappa_, params_["Equation"]["NeumannBC"].get<DataType>(),
      params_["Equation"]["DirichletMaterialNumber"].get<int>(),
      params_["Parameters"]["diffusion_1"].get<DataType>(),
      params_["Parameters"]["carrying_capacity_1"].get<DataType>(),
      params_["Parameters"]["diffusion_2"].get<DataType>(),
      params_["Parameters"]["carrying_capacity_2"].get<DataType>(), t_, a_, b_,
      theta_, dt_);

  local_asm.set_prev_time_solution(&prev_sol_);
  local_asm.set_newton_solution(&in);

  this->global_asm_.should_reset_assembly_target(true);

  this->global_asm_.assemble_vector(this->space_, local_asm, *out);

  if (!this->dirichlet_dofs_.empty()) {
    std::vector<DataType> zeros(this->dirichlet_dofs_.size(), 0.);
    out->SetValues(vec2ptr(this->dirichlet_dofs_), this->dirichlet_dofs_.size(),
                   vec2ptr(zeros));
  }
}

void FisherKolmogorovGrowth::solve_system() {
  this->kappa_ = params_["Equation"]["Kappa"].get<DataType>();
  sol_.Update();
  newton_.Solve(&sol_);
  sol_.Update();
  LOG_INFO(1, "Newton ended with residual norm " << newton_.GetResidual()
                                                 << " after " << newton_.iter()
                                                 << " iterations.");
  prev_sol_.CloneFrom(sol_);
}

void FisherKolmogorovGrowth::visualize() {

  // Setup visualization object.
  int num_sub_intervals = 1;
  CellVisualization<DataType, DIM> visu(space_, num_sub_intervals);

  // collect cell-wise data from mesh object
  const int tdim = mesh_->tdim();
  const int num_cells = mesh_->num_entities(tdim);
  std::vector<DataType> remote_index(num_cells, 0);
  std::vector<DataType> sub_domain(num_cells, 0);
  std::vector<DataType> material_number(num_cells, 0);

  // loop through all cells in the mesh
  for (mesh::EntityIterator it = mesh_->begin(tdim); it != mesh_->end(tdim);
       ++it) {
    int temp1, temp2;
    const int cell_index = it->index();
    if (DIM > 1) {
      mesh_->get_attribute_value("_remote_index_", tdim, cell_index, &temp1);
      mesh_->get_attribute_value("_sub_domain_", tdim, cell_index, &temp2);
      remote_index.at(cell_index) = temp1;
      sub_domain.at(cell_index) = temp2;
    }
    material_number.at(cell_index) =
        mesh_->get_material_number(tdim, cell_index);
  }

  // visualize finite element function corresponding to
  // coefficient vector sol_
  visu.visualize(sol_, 0, "u");

  // visualize some mesh data
  visu.visualize_cell_data(remote_index, "_remote_index_");
  visu.visualize_cell_data(sub_domain, "_sub_domain_");
  visu.visualize_cell_data(material_number, "Material Id");

  // write out data
  std::stringstream name;
  name << "fk_model_sol_";

  if (ts_ < 10)
    name << "000" << ts_;
  else if (ts_ < 100)
    name << "00" << ts_;
  else if (ts_ < 1000)
    name << "0" << ts_;
  else
    name << "" << ts_;

  VTKWriter<DataType, DIM> vtk_writer(visu, this->comm_, MASTER_RANK);
  vtk_writer.write(name.str());
}

void FisherKolmogorovGrowth::adapt() {
  if (rank_ == MASTER_RANK) {
    const int final_ref_level = params_["Mesh"]["FinalRefLevel"].get<int>(6);
    if (refinement_level_ >= final_ref_level) {
      is_done_ = true;
    } else {
      master_mesh_ = master_mesh_->refine();
      ++refinement_level_;
    }
  }

  // Broadcast information from master to slaves.
  MPI_Bcast(&refinement_level_, 1, MPI_INT, MASTER_RANK, comm_);
  MPI_Bcast(&is_done_, 1, MPI_CHAR, MASTER_RANK, comm_);

  if (!is_done_) {
    // Distribute the new mesh.
    MeshPtr local_mesh =
        partition_and_distribute(master_mesh_, MASTER_RANK, comm_);
    assert(local_mesh != 0);
    SharedVertexTable shared_verts;
    mesh_ = compute_ghost_cells(*local_mesh, comm_, shared_verts);
  }
}
