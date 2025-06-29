#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Parameter.h"
#include "Sampler.h"
#include "Scenario.h"
#include "Criterion.h"
#include "Collector.h"
#include "Simulator.h"


namespace py = pybind11;
using namespace eventide;


static std::string to_lower(std::string s) {
     std::transform(s.begin(), s.end(), s.begin(),
                    [](const unsigned char c) { return std::tolower(c); }
     );
     return s;
}

static DrawID name_to_id(const std::string& name) {
     const std::string n = to_lower(name);
     if (n == "r0") return DrawID::R0;
     if (n == "k") return DrawID::k;
     if (n == "r") return DrawID::r;
     if (n == "alpha") return DrawID::alpha;
     if (n == "theta") return DrawID::theta;
     throw py::value_error("Unknown parameter name '" + name + "'");
}


namespace eventide {
     struct PySimulator {
          std::unique_ptr<Simulator> core;
          std::vector<DataCollector*> originals; // borrowed, Python-side

          PySimulator(std::unique_ptr<Simulator> core_, std::vector<DataCollector*> originals_):
               core(std::move(core_)), originals(std::move(originals_)) {}

          void run() const {
               core->run();
               auto& clone_group = core->collectors(); // DataCollectorGroup
               const size_t n_clones = clone_group.size();
               const size_t n_orig = originals.size();

               const size_t n = std::min(n_clones, n_orig);
               for (size_t i = 0; i < n; ++i)
                    originals[i]->merge(*clone_group.at(i));
          }
     };
}

/* reorder user given Parameter objects into the fixed order
 * [R0, k, r, alpha, theta].  Accepts either a list or a dict. */
static std::vector<Parameter> reorder_params(const py::object& seq) {
     std::array<std::optional<Parameter>, 5> slot;

     auto emplace = [&](const Parameter& p) {
          slot[static_cast<int>(name_to_id(p.name))].emplace(p);
     };

     if (py::isinstance<py::dict>(seq)) {
          for (auto [fst, snd] : seq.cast<py::dict>())
               emplace(snd.cast<const Parameter&>());
     }
     else {
          for (auto p_obj : seq)
               emplace(p_obj.cast<const Parameter&>());
     }
     for (int i = 0; i < 5; ++i)
          if (!slot[i]) throw py::value_error("missing parameter");

     // build vector in final order by *construction*, not assignment
     return {*slot[0], *slot[1], *slot[2], *slot[3], *slot[4]};
}

PYBIND11_MODULE(_eventide, m) {
     m.doc() = "Branching‐process simulator";

     // Parameter
     py::class_<Parameter>(m, "Parameter")
          .def(py::init<std::string, double, double>(),
               py::arg("name"),
               py::arg("min"),
               py::arg("max")
          )
          .def_readonly("name", &Parameter::name)
          .def_readonly("min", &Parameter::min)
          .def_readonly("max", &Parameter::max)
          .def("is_fixed", &Parameter::isFixed);

     py::class_<CompiledExpression>(m, "_Expr")
          .def(py::init<std::string>());

     // LatinHypercubeSampler
     py::class_<LatinHypercubeSampler>(m, "LatinHypercubeSampler")
          .def(py::init([](const py::object& params, bool scramble) {
                    static std::vector<std::unique_ptr<RngEngine>> all_rngs; // keep alive
                    all_rngs.emplace_back(std::make_unique<RngEngine>());
                    RngEngine& rng = *all_rngs.back();

                    std::vector<Parameter> v = reorder_params(params);
                    return std::make_unique<LatinHypercubeSampler>(v, rng, scramble);
               }),
               py::arg("parameters"),
               py::arg("scramble") = true
          )
          .def("sample_block",
               [](LatinHypercubeSampler& self, const int n) {
                    return self.sampleBlock(n); // rng is stored internally
               })
          .def("parameters", [](const LatinHypercubeSampler& self) { return self.parameters(); });

     // Scenario & ChangePoints
     py::class_<ParameterChangePoint>(m, "ParameterChangePoint")
          .def(py::init([](const double time, const std::string& paramName, const CompiledExpression& expression) {
                    return ParameterChangePoint(time, name_to_id(paramName), expression);
               }),
               py::arg("time"),
               py::arg("param"),
               py::arg("expr"),
               "At <time>, set <param> to <new_value>."
          )
          .def(py::init([](const double time, const std::string& paramName) {
                    return ParameterChangePoint(time, name_to_id(paramName));
               }),
               py::arg("time"),
               py::arg("param"),
               "At <time>, restore <param> to its original draw."
          );


     py::class_<Scenario>(m, "Scenario")
          .def(py::init<std::vector<ParameterChangePoint>>(),
               py::arg("change_points")
          );

     // Criterion base + subclasses
     py::class_<Criterion, std::unique_ptr<Criterion>>(m, "Criterion");

     py::class_<OffspringCriterion, Criterion>(m, "OffspringCriterion")
          .def(py::init<int, int>(),
               py::arg("min_offspring"),
               py::arg("max_offspring")
          );

     py::class_<IntervalCriterion, Criterion>(m, "IntervalCriterion")
          .def(py::init<double, double, int, int>(),
               py::arg("t_min"),
               py::arg("t_max"),
               py::arg("min_allowed"),
               py::arg("max_allowed")
          );

     // DataCollector base + subclasses
     py::class_<DataCollector, std::unique_ptr<DataCollector>>(m, "DataCollector");

     py::class_<TimeMatrixCollector, DataCollector>(m, "_TimeMatrixCollector")
          .def(py::init<int, int>(),
               py::arg("T"),
               py::arg("cutoff_day")
          )
          .def("matrix", &TimeMatrixCollector::matrix, py::return_value_policy::reference_internal);

     py::class_<Hist1D, DataCollector>(m, "_Hist1D")
          .def(py::init<CompiledExpression, int, double, double>(),
               py::arg("expr"),
               py::arg("bins"),
               py::arg("lo"),
               py::arg("hi")
          )
          .def("histogram", &Hist1D::histogram, py::return_value_policy::reference_internal);

     py::class_<Hist2D, DataCollector>(m, "_Hist2D")
          .def(py::init<CompiledExpression, CompiledExpression, int, double, double, double, double>(),
               py::arg("expr_x"),
               py::arg("expr_y"),
               py::arg("bins"),
               py::arg("lo_x"),
               py::arg("hi_x"),
               py::arg("lo_y"),
               py::arg("hi_y")
          )
          .def("histogram", &Hist2D::histogram, py::return_value_policy::reference_internal);

     // Simulator
     py::class_<PySimulator>(m, "Simulator")
          .def(py::init([](LatinHypercubeSampler& sampler,
                           Scenario& scenario,
                           const std::vector<Criterion*>& pyCrit,
                           const std::vector<DataCollector*>& pyColl,
                           const CompiledExpression& validator,
                           int64_t numT, int chunk, int Tr, int maxC,
                           int workers) {
                    std::vector<std::unique_ptr<Criterion>> critCopies;
                    for (const auto* c : pyCrit) critCopies.emplace_back(c->clone());
                    CriterionGroup critGroup(std::move(critCopies));

                    std::vector<std::unique_ptr<DataCollector>> collCopies;
                    for (const auto* c : pyColl) collCopies.emplace_back(c->clone());
                    DataCollectorGroup collGroup(std::move(collCopies));


                    auto core = std::make_unique<Simulator>(
                         sampler, scenario,
                         critGroup, collGroup,
                         numT, chunk, Tr, maxC,
                         workers, validator);


                    return PySimulator(std::move(core), pyColl);
               }),
               py::arg("sampler"),
               py::arg("scenario"),
               py::arg("criteria"),
               py::arg("collectors"),
               py::arg("validator"),
               py::arg("num_trajectories"),
               py::arg("chunk_size"),
               py::arg("T_run"),
               py::arg("max_cases"),
               py::arg("max_workers"))
          .def("run", &PySimulator::run);
}
