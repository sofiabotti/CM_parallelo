#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>


#include <deal.II/lac/la_parallel_block_vector.h>

namespace FHN {
    using namespace dealii;

    namespace LA {
        // using namespace dealii::LinearAlgebraPETSc;
        using namespace dealii::LinearAlgebraTrilinos;
    } // namespace LA

    template<int dim>
    class FitzhughNagumo {
    public:
        FitzhughNagumo();
        void run();

    private:
        void setup_system();
        void solve_time_step();
        void output_results() const;
        void assemble_system();
        void solve_gating();
        void initial_conditions();


        void refine_mesh(const unsigned int min_grid_level,
                         const unsigned int max_grid_level);

        MPI_Comm communicator;

        ConditionalOStream pout;
        mutable TimerOutput timer;

        parallel::distributed::Triangulation<dim> triangulation;
        FE_Q<dim> fe;
        DoFHandler<dim> dof_handler;

// Figure out who are my dofs, and my locally_relevant dofs
        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;

        AffineConstraints<double> constraints;

        LA::MPI::SparseMatrix mass_matrix;
        LA::MPI::SparseMatrix laplace_matrix;
        LA::MPI::SparseMatrix system_matrix;
        LA::MPI::Vector solution;
        LA::MPI::Vector old_solution;
        LA::MPI::Vector system_rhs;
        LA::MPI::Vector rhs;

        LA::MPI::Vector locally_relevant_solution;

        const unsigned int n_gate = 1;

        std::vector<LA::MPI::Vector>    var_gating;
        std::vector<LA::MPI::Vector>    locally_relevant_gating;

        LA::MPI::Vector     I_ion;

        double time;
        double time_step;
        unsigned int timestep_number;
        bool flag_refinement;

        const double theta;
        const double sigma;
        const int T_end;

    };

    template<int dim>
    class AppliedCurrent : public Function<dim> {
    public:
        AppliedCurrent()
                :
                Function<dim>(),
                amplitude(2),
                duration(2),
                period(10000),
                end_stim(50),
                start_stim(0) {}

        virtual double value(const Point<dim> &p,
                             const unsigned int component = 0) const;

    private:
        const double amplitude;
        const double duration;
        const double period;
        const double end_stim, start_stim;
    };


    template<int dim>
    double AppliedCurrent<dim>::value(const Point<dim> &p,
                                      const unsigned int component) const {
        (void) component;
     //   Assert (component == 0, ExcIndexRange(component, 0, 1));
     //   Assert (dim == 1, ExcNotImplemented());

        const double time = this->get_time();
        const double point_within_period = (time / period - std::floor(time / period));

        if ((time >= start_stim) && (time <= end_stim) && (point_within_period * period <= duration))
            if ((p[0] < 0.05))
                return amplitude;
            else
                return 0;
        else
            return 0;

    }


  template<int dim>
  FitzhughNagumo<dim>::FitzhughNagumo ()
    :
    communicator(MPI_COMM_WORLD),
    pout(std::cout, Utilities::MPI::this_mpi_process(communicator)==0),
    timer(pout, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
    triangulation(communicator),
    fe(1),
    dof_handler(triangulation),
    time (0.0),
    time_step(1. / 500),
    timestep_number (0),
	flag_refinement(true),
    theta(1),
    sigma(0.001),
    T_end (50)
  {}


  template<int dim>
  void FitzhughNagumo<dim>::setup_system()
  {
    TimerOutput::Scope timer_section(timer, "Setup system");
    dof_handler.distribute_dofs(fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pout << std::endl
      << "==========================================="
      << std::endl
      << "Number of active cells: " << triangulation.n_active_cells()
      << std::endl
      << "Number of degrees of freedom: " << dof_handler.n_dofs()
      << std::endl
      << std::endl;

    constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    SparsityTools::distribute_sparsity_pattern(
          dsp,
          dof_handler.n_locally_owned_dofs_per_processor(),
          communicator,
          locally_relevant_dofs);

    mass_matrix.reinit(locally_owned_dofs, dsp, communicator);
    laplace_matrix.reinit(locally_owned_dofs, dsp, communicator);
    system_matrix.reinit(locally_owned_dofs, dsp, communicator);

    assemble_system();
    laplace_matrix *= sigma;

    solution.reinit(locally_owned_dofs, communicator);
    old_solution.reinit(locally_owned_dofs, communicator);
    for (unsigned int i = 0; i < n_gate; ++i){
        LA::MPI::Vector gate;
        gate.reinit(locally_owned_dofs, communicator);
        var_gating.push_back(gate);
    }
    I_ion.reinit(locally_owned_dofs, communicator);
    system_rhs.reinit(locally_owned_dofs, communicator);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     communicator);
    for (unsigned int i = 0; i < n_gate; ++i){
         LA::MPI::Vector lrgate;
         lrgate.reinit(locally_owned_dofs, locally_relevant_dofs, communicator);
         var_gating.push_back(lrgate);
    }
    locally_relevant_gating = var_gating;
  }

template<int dim>
void FitzhughNagumo<dim>::assemble_system() {
    TimerOutput::Scope timer_section(timer, "assemble system");

    AppliedCurrent<dim> rhs_function;
    rhs_function.set_time(time);
    rhs.reinit(locally_owned_dofs, communicator);

    QGauss<dim> quadrature_formula(fe.degree + 1);
    MeshWorker::ScratchData<dim> scratch(fe,
                                         quadrature_formula,
                                         update_quadrature_points |
                                         update_values | update_gradients |
                                         update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    std::vector<double> local_I_ion(n_q_points);

    MeshWorker::CopyData<2, 1, 1> copy_data(dofs_per_cell);
   auto worker = [&](const decltype(dof_handler.begin_active()) &cell,
                 MeshWorker::ScratchData<dim> &scratch,
                 MeshWorker::CopyData<2, 1, 1> &copy_data) {
   auto &fe_values = scratch.reinit(cell);

       copy_data.matrices[0] = 0;
       copy_data.matrices[1] = 0;
       copy_data.vectors[0] = 0;

       for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    copy_data.matrices[0](i, j) +=
                            (fe_values.shape_value(i, q_index) *
                             fe_values.shape_value(j, q_index) * fe_values.JxW(q_index));

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    copy_data.matrices[1](i, j) +=
                            (fe_values.shape_grad(i, q_index) *
                             fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));


           for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const double I_app = rhs_function.value(fe_values.quadrature_point(q_index), 0);
                //const double I_app = rhs_function.value(q_points[q_index]);
                copy_data.vectors[0](i) +=
                        (fe_values.shape_value(i, q_index) *
 //                        (I_app + local_I_ion[q_index]) * fe_values.JxW(q_index));
                         (I_app) * fe_values.JxW(q_index));
           }
       }
        cell->get_dof_indices(copy_data.local_dof_indices[0]);
    };

    auto copier = [&](const MeshWorker::CopyData<2, 1, 1> &copy_data) {
        constraints.distribute_local_to_global(copy_data.matrices[0],
                                               copy_data.vectors[0],
                                               copy_data.local_dof_indices[0],
                                               mass_matrix,
                                               rhs);
        constraints.distribute_local_to_global(copy_data.matrices[1],
                                               copy_data.local_dof_indices[0],
                                               laplace_matrix);
    };


    using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
            CellFilter(IteratorFilters::LocallyOwnedCell(), dof_handler.begin_active()),
            CellFilter(IteratorFilters::LocallyOwnedCell(), dof_handler.end()),
            worker,
            copier,
            scratch,
            copy_data);

    mass_matrix.compress(VectorOperation::add);
    laplace_matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
}

  template<int dim>
  void FitzhughNagumo<dim>::solve_time_step(){

  TimerOutput::Scope timer_section(timer, "Solve system");
  SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
  LA::SolverCG cg(solver_control);

  LA::MPI::PreconditionAMG::AdditionalData data;
  LA::MPI::PreconditionAMG amg;
  amg.initialize(system_matrix);

  cg.solve(system_matrix, solution, system_rhs, amg);
  constraints.distribute(solution);
  for (unsigned int i = 0; i < n_gate; ++i)
      constraints.distribute(var_gating[i]);
  locally_relevant_solution = solution;

  /*    std::cout << "     " << solver_control.last_step()
              << " CG iterations." << std::endl; */
  }

  template<int dim>
  void FitzhughNagumo<dim>::solve_gating() {
      TimerOutput::Scope timer_section(timer, "Solve gating");

      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned()) {
              for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  auto v = solution[i];
                  auto m = var_gating[0](i);
                  m = m + time_step * 0.1 * (v - 0.25 * m);
                  var_gating[0](i) = m;
                  I_ion[i] = 5 * v * (v - 0.1) * (1 - v) - m;
              }
              cell->get_dof_indices(local_dof_indices);
          }
      locally_relevant_gating = var_gating;
  }

    template<int dim>
    void FitzhughNagumo<dim>::initial_conditions() {
        TimerOutput::Scope timer_section(timer, "initial conditions");
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        for (const auto &cell : dof_handler.active_cell_iterators())
            if (cell->is_locally_owned()) {
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    old_solution[i] = 0;
                    var_gating[0](i) = 0;
                }
                cell->get_dof_indices(local_dof_indices);
            }
        locally_relevant_gating = var_gating;
    }
  template<int dim>
  void FitzhughNagumo<dim>::output_results() const
  {
    TimerOutput::Scope  timer_section(timer, "output results");

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "solution");

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    const std::string filename = "solution-"
                               + Utilities::int_to_string(timestep_number, 3) +
                               ".vtu";
    data_out.write_vtu_in_parallel(filename.c_str(), communicator);
    data_out.build_patches();
  }

  template <int dim>
  void FitzhughNagumo<dim>::refine_mesh (const unsigned int min_grid_level,
                                       const unsigned int max_grid_level){

    TimerOutput::Scope  timer_section(timer, "refine mesh");

    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(fe.degree+1),
                                        {},
                                        locally_relevant_solution,
                                        estimated_error_per_cell);

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                       estimated_error_per_cell,
                                                       0.1, 0.);

    if (triangulation.n_levels() > max_grid_level)
      for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active(max_grid_level);
           cell != triangulation.end(); ++cell)
        cell->clear_refine_flag ();
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active(min_grid_level);
         cell != triangulation.end_active(min_grid_level); ++cell)
      cell->clear_coarsen_flag ();

    LA::MPI::BlockVector in;
    LA::MPI::BlockVector out;

    in.reinit(n_gate + 1);
    in.block(0).reinit(locally_owned_dofs, locally_relevant_dofs, communicator);
    for (unsigned int i = 1; i < n_gate + 1; ++i)
        in.block(i).reinit(locally_owned_dofs,locally_relevant_dofs, communicator);

    in.block(0) = locally_relevant_solution;
    for (unsigned int i = 0; i < n_gate; ++i)
        in.block(i + 1) = locally_relevant_gating[i];

    triangulation.prepare_coarsening_and_refinement();
    parallel::distributed::SolutionTransfer<dim, LA::MPI::BlockVector> solution_trans(dof_handler);
    solution_trans.prepare_for_coarsening_and_refinement(in);
    triangulation.execute_coarsening_and_refinement();

    setup_system();

    out.reinit(n_gate + 1);
    out.block(0).reinit(locally_owned_dofs, communicator);
    for (unsigned int i = 1; i < n_gate + 1; ++i)
        out.block(i).reinit(locally_owned_dofs, communicator);

    out.block(0) = solution;
    for (unsigned int i = 0; i < n_gate; ++i)
        out.block(i + 1) = var_gating[i];

    solution_trans.interpolate(out);
    constraints.distribute(out);

    solution = out.block(0);
    for (unsigned int i = 0; i < n_gate; ++i)
        var_gating[i] = out.block(i + 1);

   /* std::vector<LA::MPI::Vector> in;
    std::vector<LA::MPI::Vector> out;

    in.push_back(locally_relevant_solution);
    for (unsigned int i = 1; i < n_gate + 1; ++i)
        in.push_back(locally_relevant_gating[i]);

    triangulation.prepare_coarsening_and_refinement();
    parallel::distributed::SolutionTransfer<dim, std::vector<LA::MPI::Vector>> solution_trans(dof_handler);
    solution_trans.prepare_for_coarsening_and_refinement(in);
    triangulation.execute_coarsening_and_refinement();

    setup_system();

    out.push_back(solution);
    for (unsigned int i = 1; i < n_gate + 1; ++i)
        out.push_back(var_gating[i]);

    solution_trans.interpolate(out);
    constraints.distribute(out);

    solution = out[0];
    for (unsigned int i = 0; i < n_gate; ++i)
        var_gating[i] = out[i + 1];
        */

  }


  template<int dim>
  void FitzhughNagumo<dim>::run()
  {
    const unsigned int initial_global_refinement = 7;
    const unsigned int n_adaptive_pre_refinement_steps = 4;
    unsigned int pre_refinement_step = 0;

    GridGenerator::hyper_cube (triangulation, 0, 1);
    triangulation.refine_global (initial_global_refinement);

    setup_system();

    LA::MPI::Vector tmp;
    LA::MPI::Vector forcing_terms;

    start_time_iteration:

    tmp.reinit (solution, communicator);
    forcing_terms.reinit (solution, communicator);

    initial_conditions();
    solution = old_solution;

    output_results();

    while (time <= T_end){

//      solve_gating();
      time += time_step;
      ++timestep_number;

      pout << "Time step " << timestep_number << " at t=" << time
                << std::endl;

      mass_matrix.vmult(system_rhs, old_solution);

      laplace_matrix.vmult(tmp, old_solution);
      system_rhs.add(-(1 - theta) * time_step, tmp);

	  assemble_system();

	  forcing_terms = rhs;
      forcing_terms *= time_step * theta;

	  assemble_system();

        forcing_terms.add(time_step * (1 - theta), rhs);

        system_rhs += forcing_terms;

        system_matrix.copy_from(mass_matrix);
        system_matrix.add(theta * time_step, laplace_matrix);

        solve_time_step();

        Vector<double> stampa;
        stampa = solution;

	  if (timestep_number % 15 == 0)
		output_results();

      if (flag_refinement){
        if ((timestep_number == 0) &&
            (pre_refinement_step < n_adaptive_pre_refinement_steps))
          {
            refine_mesh (initial_global_refinement,
                         initial_global_refinement + n_adaptive_pre_refinement_steps);
            ++pre_refinement_step;

            tmp.reinit (solution, communicator);
            forcing_terms.reinit (solution, communicator);

            std::cout << std::endl;

            goto start_time_iteration;
          }
        else if ((timestep_number > 0) && (timestep_number % 5 == 0))
          {
            refine_mesh (initial_global_refinement,
                         initial_global_refinement + n_adaptive_pre_refinement_steps);
            tmp.reinit (solution, communicator);
            forcing_terms.reinit (solution, communicator);
          }
      }
        old_solution = solution;
    }
  }
}

int main(int argc, char **argv)
{
    using namespace dealii;
    using namespace FHN;


    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;

    FitzhughNagumo<2> fitzhugh_nagumo_solver;
    fitzhugh_nagumo_solver.run();

    return 0;

}
