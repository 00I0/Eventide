// CompiledExpression.cpp
#include "CompiledExpression.h"


using namespace eventide;

static std::string make_temp_name(std::string const& suffix) {
    static int counter = 0;
    const auto tmp = std::filesystem::temp_directory_path() / ("eventide_expr_" + std::to_string(++counter) + suffix);
    return tmp.string();
}

struct CompiledExpression::Impl {
    using FnPtr = double (*)(double, double, double, double, double);
    void* handle = nullptr;
    FnPtr fn = nullptr;
    std::string libpath;


    ~Impl() {
        if (handle) {
            dlclose(handle);
            handle = nullptr;
        }
        if (!libpath.empty()) {
            std::filesystem::remove(libpath);
            libpath.clear();
        }
    }
};

CompiledExpression::CompiledExpression(const std::string& expr): impl_(std::make_shared<Impl>()) {
    // 1) Emit temporary .cpp
    std::string cpp_path = make_temp_name(".cpp");
    std::ofstream out(cpp_path);
    out << "extern \"C\" double f(double R0, double k, double r, double alpha, double theta) {\n"
        << "    return " << expr << ";\n"
        << "}\n";
    out.close();

    // 2) Compile to shared library
    impl_->libpath = make_temp_name(
#ifdef __APPLE__
        ".dylib"
#else
            ".so"
#endif
    );
    std::ostringstream cmd;
    cmd << "clang++ -O3 -std=c++17 -fPIC -shared -march=native -ffast-math " << cpp_path << " -o " << impl_->libpath;
    if (std::system(cmd.str().c_str()) != 0)
        throw std::runtime_error("compile failed: " + cmd.str());


    // 3) dlopen & dlsym
    impl_->handle = dlopen(impl_->libpath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!impl_->handle) throw std::runtime_error(dlerror());
    impl_->fn = reinterpret_cast<Impl::FnPtr>(dlsym(impl_->handle, "f"));
    if (!impl_->fn) throw std::runtime_error(dlerror());
}


double CompiledExpression::eval(const Draw& draw) const {
    return impl_->fn(draw.R0, draw.k, draw.r, draw.alpha, draw.theta);
}

