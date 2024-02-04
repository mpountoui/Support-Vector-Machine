#include <iostream>
#include <cstdio>
#include <cassert>
#include "Vectors.hpp"
#include "Sequential_minimal_optimization.hpp"

/* ----------------------------------------------------------------------------------------- */

double epsilon = 1e-3;

/* ----------------------------------------------------------------------------------------- */

DualProblem::DualProblem(SVM& SVM_in)
    :
    m_SVM(SVM_in)
{}

/* ----------------------------------------------------------------------------------------- */

std::pair<double, double> DualProblem::EndsOfLine(double a_1, double a_2, bool flag) const
{
    if (flag)
    {
        return { std::max(.0, a_2 + a_1 - m_SVM.m_C), std::min(m_SVM.m_C, a_2 + a_1) };
    }

    return { std::max(.0, a_2 - a_1), std::min(m_SVM.m_C, m_SVM.m_C + a_2 - a_1) };
}

/* ----------------------------------------------------------------------------------------- */

void DualProblem::UpdateLagrangeMultipliers(size_t i, size_t k, double a_1, double a_2) const
{
    m_SVM.m_a[i] = a_1;
    m_SVM.m_a[k] = a_2;
}

/* ----------------------------------------------------------------------------------------- */

bool DualProblem::UpdateErros()
{
    std::unordered_map<size_t, double> Errors;

    for(size_t q = 0; q < m_SVM.m_a.size(); ++q)
    {
        double a = m_SVM.m_a[q];
        if( a > epsilon && a < m_SVM.m_C - epsilon )
        {
            double E = m_SVM(m_SVM.m_x->operator[](q)) - m_SVM.m_y->operator[](q);
            
            auto check = Errors.insert( {q, E} );
            if( !check.second )
            {
                assert(false); 
                return false;
            }
        }
    }

    m_Errors = std::move(Errors);
    return true;
}

/* ----------------------------------------------------------------------------------------- */

double DualProblem::GetError(size_t key) const
{
    auto it = m_Errors.find(key);
    if( it != m_Errors.cend() )
    {
        return it->second;
    }

    return m_SVM(m_SVM.m_x->operator[](key)) - m_SVM.m_y->operator[](key);
}

/* ----------------------------------------------------------------------------------------- */

size_t DualProblem::ChooseSecondLagrangeMultiplier(double E_2) const
{
    auto it = m_Errors.cbegin();

    if( E_2 > 0 )
    {
        it = std::min_element( m_Errors.cbegin(), m_Errors.cend(), [] (auto const& p1, auto const& p2) 
        {
            return p1.second < p2.second;
        });

    }
    else
    {
        it = std::max_element( m_Errors.cbegin(), m_Errors.cend(), [] (auto const& p1, auto const& p2) 
        {
            return p1.second < p2.second;
        });
    }

    return it->first;
}

/* ----------------------------------------------------------------------------------------- */

bool DualProblem::PerformStep(size_t i, size_t k, double E_2)
{
    if(i == k){ /*assert(false);*/ return false; }
    Kernel& K        = *(m_SVM.m_Kernel);
    Vector const x_1 = m_SVM.m_x->operator[](i);
    Vector const x_2 = m_SVM.m_x->operator[](k);
    int y_1          = m_SVM.m_y->operator[](i);
    int y_2          = m_SVM.m_y->operator[](k);
    int s            = y_1 * y_2;
    double a_1       = m_SVM.m_a[i];
    double a_2       = m_SVM.m_a[k];
    double E_1       = GetError(i);
    double k_11      = K( x_1, x_1 );
    double k_12      = K( x_1, x_2 );
    double k_22      = K( x_2, x_2 );
    double htta      = k_11 + k_22 - 2*k_12;

    auto [L_2, H_2] = EndsOfLine(a_1, a_2, s > 0);
    if( H_2 < L_2 || std::fabs(L_2 - H_2) < epsilon ){ assert(false); return false; }
    double L_1 = a_1 + s * (a_2 - L_2);
    double H_1 = a_1 + s * (a_2 - H_2);

    if(htta > 0){
        a_2 = a_2 + y_2*(E_1-E_2)/htta;
        a_2 = std::min( std::max(a_2, L_2) , H_2 );
    }else{
        double f_1     = y_1*(E_1 + m_SVM.m_b) -     a_1 * k_11 - s * a_2 * k_12;
        double f_2     = y_2*(E_2 + m_SVM.m_b) - s * a_1 * k_12 -     a_2 * k_22;
        double ObjFunL = (L_1 * f_1) + (L_2 * f_2) + (0.5 * L_1 * L_1 * k_11) + (0.5 * L_2 * L_2 * k_22) + (s * L_2 * L_1 * k_12);
        double ObjFunH = (H_1 * f_1) + (H_2 * f_2) + (0.5 * H_1 * H_1 * k_11) + (0.5 * H_2 * H_2 * k_22) + (s * H_2 * H_1 * k_12);
        if      (ObjFunL < ObjFunH - epsilon){ a_2 = L_2; }
        else if (ObjFunL > ObjFunH + epsilon){ a_2 = H_2; }
    }

    if ( std::fabs( a_2 - m_SVM.m_a[k] ) < epsilon * (a_2 + m_SVM.m_a[k] + epsilon) ){ return false; }
    a_1 += s * (m_SVM.m_a[k] - a_2);

    double b_1 = E_1 + y_1 * (a_1 - m_SVM.m_a[i]) * k_11 + y_2 * (a_2 - m_SVM.m_a[k]) * k_12 + m_SVM.m_b;
    double b_2 = E_2 + y_1 * (a_1 - m_SVM.m_a[i]) * k_12 + y_2 * (a_2 - m_SVM.m_a[k]) * k_22 + m_SVM.m_b;
    if(      std::fabs(a_1 - L_1) > epsilon && std::fabs(a_1 - H_1) > epsilon ){ m_SVM.m_b = b_1;               }
    else if( std::fabs(a_2 - L_2) > epsilon && std::fabs(a_2 - H_2) > epsilon ){ m_SVM.m_b = b_2;               }
    else                                                                       { m_SVM.m_b = (b_2 + b_1) / 2.0; }

    K.W_UpdateIfLinearKernel(y_1, a_1 - m_SVM.m_a[i], x_1, y_2, a_2 - m_SVM.m_a[k], x_2);
    UpdateLagrangeMultipliers(i, k, a_1, a_2);
    if( !UpdateErros() ){ assert(false); return false; }

    return true;
}

/* ----------------------------------------------------------------------------------------- */

bool DualProblem::ExamineExample(size_t k)
{
    int y_2      = m_SVM.m_y->operator[](k);
    double a_2   = m_SVM.m_a[k];
    double E_2   = GetError(k);
    double check = E_2 * y_2;

    if( ( check < - epsilon && a_2 < m_SVM.m_C - epsilon ) || ( check > epsilon && a_2 > epsilon) )
    {
        if ( m_Errors.size() > 1 )
        {
            size_t i = ChooseSecondLagrangeMultiplier(E_2);
            if( PerformStep(i, k, E_2) ){ return true; }
        }

        for(auto p : m_Errors)
        {
            if( PerformStep(p.first, k, E_2) ) { return true; }
        }
        
        size_t StartPosition = rand() % m_SVM.m_a.size();
        size_t SIZE = m_SVM.m_a.size();
        for(size_t i = StartPosition; i < SIZE; ++i)
        {
            if( PerformStep(i, k, E_2) ) { return true; }
        }
        
        for(int i = StartPosition - 1; i >= 0 ; --i)
        {
            if( PerformStep(i, k, E_2) ) { return true; }
        }
    }

    /* KKT Conditions Satisfied */
    return false;
}

/* ----------------------------------------------------------------------------------------- */

bool DualProblem::Solve()
{
    size_t NumberChanged = 0;
    bool ExamineAll = true;
    
    while( NumberChanged > 0 || ExamineAll )
    {
        NumberChanged = 0;
        if( ExamineAll )
        {
            for(int k = 0; k < m_SVM.m_a.size(); ++k)
            {
                if( ExamineExample(k) ){ ++NumberChanged; }
            }
        }
        else
        {
            std::vector<size_t> Keys;
            Keys.reserve( m_Errors.size() );
            for( auto p : m_Errors )
            {
                Keys.push_back(p.first);
            }
            for(int k = 0; k < Keys.size(); ++k)
            {
                if( ExamineExample( Keys[k] ) ){ ++NumberChanged; }
            }
        }

        if (ExamineAll)
        {
            ExamineAll = false;
        }
        else if (NumberChanged == 0)
        {
            ExamineAll = true;
        }
    }

    return true;
}
