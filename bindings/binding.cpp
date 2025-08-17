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
                    [](const unsigned char c) { return std::tolower(c); });
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
          std::vector<std::shared_ptr<DataCollector>> originals; // borrowed, Python-side

          PySimulator(const std::shared_ptr<Sampler>& sampler,
                      const std::shared_ptr<Scenario>& scenario,
                      const std::vector<std::shared_ptr<Criterion>>& criteria,
                      const std::vector<std::shared_ptr<DataCollector>>& collectors,
                      CompiledExpression validator,
                      int64_t numT,
                      int chunk,
                      int Tr,
                      int maxC,
                      int workers)
               : originals(collectors) {
               CriterionGroup critGroup(criteria);
               DataCollectorGroup collGroup(originals);

               core = std::make_unique<Simulator>(
                    *sampler, *scenario,
                    critGroup, collGroup,
                    numT, chunk, Tr, maxC, workers,
                    validator
               );
          }

          void run() const {
               core->run();
               auto& collectors = core->collectors(); // DataCollectorGroup
               const size_t n = std::min(collectors.size(), originals.size());
               for (size_t i = 0; i < n; ++i)
                    originals[i]->merge(*collectors.at(i));
          }
     };
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

     py::class_<CompiledExpression>(m, "CompiledExpression")
          .def(py::init<std::string>());


     py::class_<Sampler, std::shared_ptr<Sampler>>(m, "Draw");

     // LatinHypercubeSampler
     py::class_<LatinHypercubeSampler, Sampler, std::shared_ptr<LatinHypercubeSampler>>(m, "LatinHypercubeSampler")
          .def(py::init([](const std::vector<Parameter>& vec, bool scramble) {
                    RngEngine rng;
                    return std::make_shared<LatinHypercubeSampler>(vec, rng, scramble);
               }),
               py::arg("parameters"),
               py::arg("scramble") = true
          );


     py::class_<PreselectedSampler, Sampler, std::shared_ptr<PreselectedSampler>>(m, "PreselectedSampler")
          .def(py::init([](const std::vector<std::array<double, 5>>& rawDraws, int maxTrials) {
               std::vector<Draw> draws;
               draws.reserve(rawDraws.size());
               for (const auto& array : rawDraws)
                    draws.emplace_back(Draw{
                         array[static_cast<int>(DrawID::R0)],
                         array[static_cast<int>(DrawID::k)],
                         array[static_cast<int>(DrawID::r)],
                         array[static_cast<int>(DrawID::alpha)],
                         array[static_cast<int>(DrawID::theta)]
                    });
               return std::make_shared<PreselectedSampler>(draws, maxTrials);
          }));

     // Scenario & ChangePoints
     py::class_<ParameterChangePoint, std::shared_ptr<ParameterChangePoint>>(m, "ParameterChangePoint")
          .def(py::init([](const double time, const std::string& paramName, const CompiledExpression& expression) {
                    return std::make_shared<ParameterChangePoint>(time, name_to_id(paramName), expression);
               }),
               py::arg("time"),
               py::arg("param"),
               py::arg("expr"),
               "At <time>, set <param> to <new_value>."
          )
          .def(py::init([](const double time, const std::string& paramName) {
                    return std::make_shared<ParameterChangePoint>(time, name_to_id(paramName));
               }),
               py::arg("time"),
               py::arg("param"),
               "At <time>, restore <param> to its original draw."
          );


     py::class_<Scenario, std::shared_ptr<Scenario>>(m, "Scenario")
          .def(py::init<std::vector<ParameterChangePoint>>(),
               py::arg("change_points")
          );

     // Criterion base + subclasses
     py::class_<Criterion, std::shared_ptr<Criterion>>(m, "Criterion");

     py::class_<OffspringCriterion, Criterion, std::shared_ptr<OffspringCriterion>>(m, "OffspringCriterion")
          .def(py::init<int, int>(),
               py::arg("min_offspring"),
               py::arg("max_offspring")
          );

     py::class_<IntervalCriterion, Criterion, std::shared_ptr<IntervalCriterion>>(m, "IntervalCriterion")
          .def(py::init<double, double, int, int>(),
               py::arg("t_min"),
               py::arg("t_max"),
               py::arg("min_allowed"),
               py::arg("max_allowed")
          );

     // DataCollector base + subclasses
     py::class_<DataCollector, std::shared_ptr<DataCollector>>(m, "DataCollector");

     py::class_<TimeMatrixCollector, DataCollector, std::shared_ptr<TimeMatrixCollector>>(m, "TimeMatrixCollector")
          .def(py::init<int, int>(),
               py::arg("T"),
               py::arg("cutoff_day")
          )
          .def("matrix", &TimeMatrixCollector::matrix, py::return_value_policy::reference_internal);

     py::class_<Hist1D, DataCollector, std::shared_ptr<Hist1D>>(m, "Hist1D")
          .def(py::init<CompiledExpression, int, double, double>(),
               py::arg("expr"),
               py::arg("bins"),
               py::arg("lo"),
               py::arg("hi")
          )
          .def("histogram", &Hist1D::histogram, py::return_value_policy::reference_internal);

     py::class_<Hist2D, DataCollector, std::shared_ptr<Hist2D>>(m, "Hist2D")
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

     py::class_<DrawCollector, DataCollector, std::shared_ptr<DrawCollector>>(m, "DrawCollector")
          .def(py::init<>())
          .def("draws", &DrawCollector::draws, py::return_value_policy::reference_internal);

     py::class_<ActiveSetSizeCollector, DataCollector, std::shared_ptr<ActiveSetSizeCollector>>(
               m, "ActiveSetSizeCollector")
          .def(py::init<double>(), py::arg("collection_time"))
          .def("active_set_sizes", &ActiveSetSizeCollector::activeSetSizes,
               py::return_value_policy::reference_internal);


     py::class_<InfectionTimeCollector, DataCollector, std::shared_ptr<InfectionTimeCollector>>(
               m, "InfectionTimeCollector")
          .def(py::init<>())
          .def("infection_times", &InfectionTimeCollector::infectionTimes, py::return_value_policy::reference_internal);

     // Simulator
     py::class_<PySimulator, std::shared_ptr<PySimulator>>(m, "PySimulator")
          .def(py::init<
                    std::shared_ptr<Sampler>,
                    std::shared_ptr<Scenario>,
                    std::vector<std::shared_ptr<Criterion>>,
                    std::vector<std::shared_ptr<DataCollector>>,
                    CompiledExpression,
                    int64_t, int, int, int, int // numT, chunk, Tr, maxC, workers
               >(),
               py::arg("sampler"),
               py::arg("scenario"),
               py::arg("criteria"),
               py::arg("collectors"),
               py::arg("validator"),
               py::arg("num_trajectories"),
               py::arg("chunk_size"),
               py::arg("T_run"),
               py::arg("max_cases"),
               py::arg("max_workers"),

               // keep the Python‐side objects alive as long as this PySimulator lives:
               py::keep_alive<1, 2>(), // sampler
               py::keep_alive<1, 3>(), // scenario
               py::keep_alive<1, 4>(), // criteria list
               py::keep_alive<1, 5>() //  collectors' list
          )
          .def("run", &PySimulator::run);
}
