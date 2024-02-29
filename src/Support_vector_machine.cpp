#include <vector>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include "Support_vector_machine.hpp"
#include "Sequential_minimal_optimization.hpp"

/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ---------------------------------------- SVM -------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */

SVM::SVM(KernelType kernel_type, double C, size_t degree, double gamma, double Coeff)
    :
    m_C(C)
{
    m_Kernel = KernelFactory(kernel_type, degree, gamma, Coeff);
}

/* ----------------------------------------------------------------------------------------- */

SVM::SVM(SVM const& other)
{
    *this = other;
}

/* ----------------------------------------------------------------------------------------- */
   
SVM::SVM(SVM&& other)
{
    *this = std::move(other);
}

/* ----------------------------------------------------------------------------------------- */
   
SVM& SVM::operator=(SVM const& other)
{
    if(this != &other)
    {
        this->m_b      = other.m_b;
        this->m_C      = other.m_C;
        this->m_a      = other.m_a;
        this->m_x      = other.m_x;
        this->m_y      = other.m_y;
        this->m_Kernel = other.m_Kernel->clone();
    }

    return *this;
}

/* ----------------------------------------------------------------------------------------- */

SVM& SVM::operator=(SVM&& other)
{
        if(this != &other)
    {
        this->m_b = other.m_b;
        this->m_C = other.m_C;
        this->m_a = std::move(other.m_a);
        this->m_x = other.m_x;
        this->m_y = other.m_y;
        this->m_Kernel = other.m_Kernel;

        other.m_a.clear();
        other.m_Kernel = nullptr;
    }

    return *this;
}

/* ----------------------------------------------------------------------------------------- */

SVM::~SVM()
{
    delete m_Kernel;
}

/* ----------------------------------------------------------------------------------------- */

std::vector<double> const& SVM::LagrangeMultipliers() const
{
    return m_a;
}

/* ----------------------------------------------------------------------------------------- */

double SVM::operator()(Vector const& X) const
{
    double ret = 0.0;

    for(size_t i = 0; i < m_a.size(); ++i)
    {
        ret += m_y->operator[](i) * m_a[i] * m_Kernel->operator()( m_x->operator[](i), X );
    }

    return ret - m_b;
}

/* ----------------------------------------------------------------------------------------- */

void SVM::Fit(std::vector<Vector> const& x, std::vector<int> const& y)
{
    if( x.size() != y.size() ){ assert(false); return; }

    m_x = &x;
    m_y = &y;

    m_a = std::vector<double>(x.size(), 0.5);
    m_Kernel->W_InitializeIfLinearKernel(y, m_a, x);
    DualProblem(*this).Solve();
    m_Kernel->SaveSupportVectorsIfNonLinear(y, m_a, x);

    m_x = nullptr;
    m_y = nullptr;
}

/* ----------------------------------------------------------------------------------------- */

int SVM::Predict(Vector const& x) const
{
    double ret = m_Kernel->Predict(x, m_b);
    return ret > 0 ? 1 : -1;
}

/* ----------------------------------------------------------------------------------------- */

std::vector<int> SVM::Predict(std::vector<Vector> const& x_vec) const
{
    std::vector<int> y_vec;
    y_vec.reserve(x_vec.size());

    for(auto const& x : x_vec)
    {
        double ret = m_Kernel->Predict(x, m_b);

        if( ret > 0 )
        {
            y_vec.push_back(1);
        }
        else
        {
            y_vec.push_back(-1);
        }
    }
    return y_vec;
}

/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ------------------------------------ MultiClassSVM -------------------------------------- */
/* ----------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------- */

MultiClassSVM::MultiClassSVM(KernelType kernel_type, double C, size_t degree, double gamma, double Coeff)
    :
    m_kernel_type(kernel_type),
    m_degree(degree),
    m_gamma(gamma),
    m_Coef(Coeff),
    m_C(C)
{}

/* ----------------------------------------------------------------------------------------- */

std::vector<double> const MultiClassSVM::LagrangeMultipliers() const
{
    return {};
}

/* ----------------------------------------------------------------------------------------- */

std::vector<DataSet> MultiClassSVM::PrepareData(std::vector<Vector> const& x, std::vector<int> const& y) const
{
    std::vector<int> y_unique = y;
    std::sort(y_unique.begin(), y_unique.end());
    auto it = std::unique(y_unique.begin(), y_unique.end());
    y_unique.erase(it, y_unique.end());

    std::vector<std::pair<int, int>> combinations;
    for(size_t i = 0; i < y_unique.size(); ++i)
    {
        for(size_t k = i+1; k < y_unique.size(); ++k)
        {
            combinations.push_back( std::make_pair( y_unique[i], y_unique[k] ) );
        }
    }

    std::vector<DataSet> OneVsOneData(combinations.size());
    for(size_t k = 0; k < combinations.size(); ++k)
    {
        OneVsOneData[k].label_1 = combinations[k].first;
        OneVsOneData[k].label_2 = combinations[k].second;
    }

    for(size_t i = 0; i < y.size(); ++i)
    {
        for(size_t k = 0; k < combinations.size(); ++k)
        {
            if(y[i] == combinations[k].first || y[i] == combinations[k].second)
            {
                OneVsOneData[k].x.push_back(x[i]);
                OneVsOneData[k].y.push_back( y[i] == combinations[k].first ? -1 : 1 );
            }
        }
    }

    return OneVsOneData;
}

/* ----------------------------------------------------------------------------------------- */

void MultiClassSVM::Fit(std::vector<Vector> const& x, std::vector<int> const& y)
{
    if(y.size() != x.size()){ assert(false); return; }
    
    m_Models.clear();

    auto OneVsOneData = PrepareData(x, y);

    m_Models.reserve(OneVsOneData.size());
    for(size_t i = 0; i < OneVsOneData.size(); ++i)
    {
        std::cout << "Model " << i << " Started" << std::endl;
        auto const& data = OneVsOneData[i];
        m_Models.emplace_back( SVM(m_kernel_type, m_C, m_degree, m_gamma, m_Coef), data.label_1, data.label_2 );
        std::get<0>(m_Models.back()).Fit(data.x, data.y);
        std::cout << "Model " << i << " Finished" << std::endl << std::endl;
    }
}

/* ----------------------------------------------------------------------------------------- */

std::vector<int> MultiClassSVM::Predict(std::vector<Vector> const& x_vec) const
{
    std::vector<int> y_vec;
    y_vec.reserve(x_vec.size());

    for(auto const& x : x_vec)
    {
        std::unordered_map<int, size_t> LabelsCounter;
        for(auto const& model : m_Models)
        {
            int label_1 = std::get<1>(model);
            int label_2 = std::get<2>(model);
            if( LabelsCounter.find(label_1) == LabelsCounter.cend() ){ LabelsCounter.insert({label_1, 0}); }
            if( LabelsCounter.find(label_2) == LabelsCounter.cend() ){ LabelsCounter.insert({label_2, 0}); }
            std::get<0>(model).Predict(x) == 1 ? LabelsCounter[label_2]++ :
                                                 LabelsCounter[label_1]++ ;
        }
        
        auto it = std::max_element(std::begin(LabelsCounter), std::end(LabelsCounter), [] (auto const& p1, auto const & p2) 
        {
            return p1.second < p2.second;
        });

        std::vector<int> Labels;
        for(auto pair : LabelsCounter)
        {
            if(pair.second == it->second)
            {
                Labels.push_back(pair.first);
            }
        }
        std::sort(Labels.begin(), Labels.end());
        y_vec.push_back(Labels.front());
    }

    return y_vec;
}
