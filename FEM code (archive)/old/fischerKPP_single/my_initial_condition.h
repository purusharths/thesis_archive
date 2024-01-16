#ifndef MY_INITIAL_CONDITION_H
#define MY_INITIAL_CONDITION_H

class InitialConditionAssembler : private AssemblyAssistant<DIM, double>
{

public:
    InitialConditionAssembler() {}

    void set_parameters(DataType mean, DataType variance, bool normed)
    {
        mean_ = mean;
        variance_ = variance;
        normed_ = normed;
    }

    void operator()(const Element<double, DIM> &element,
                    const Quadrature<double> &quadrature, LocalVector &lv)
    {
        AssemblyAssistant<DIM, double>::initialize_for_element(element, quadrature,
                                                               false);

        const int num_q = num_quadrature_points();
        DataType mat_num_ = element.get_cell().get_material_number();

        for (int q = 0; q < num_q; ++q)
        {
            const double wq = w(q);
            const double dJ = std::abs(detJ(q));
            const size_t num_dof = this->num_dofs_total();

            for (int i = 0; i < num_dof; ++i)
            {
                for (int var = 0; var < 1; ++var)
                {
                    lv[i] +=
                        wq * ((1 / (2 * M_PI)) * std::sqrt(variance_)) *
                        exp(-(x(q)[0] - mean_) * (x(q)[0] - mean_) / variance_ -
                            (x(q)[1] - mean_) * (x(q)[1] - mean_) / variance_) *
                        dJ;
                }
            } // var
        }     // loop for q
    }

private:
    DataType mean_, variance_;
    bool normed_;
};

class MassMatrixAssemblerRD : private AssemblyAssistant<DIM, double>
{

public:
    MassMatrixAssemblerRD() {}

    void operator()(const Element<double, DIM> &element,
                    const Quadrature<double> &quadrature, LocalMatrix &lm)
    {
        AssemblyAssistant<DIM, double>::initialize_for_element(element, quadrature,
                                                               false);

        const int num_q = num_quadrature_points();

        for (int q = 0; q < num_q; ++q)
        {
            const double wq = w(q);
            const double dJ = std::abs(detJ(q));

            for (int var = 0; var < 1; ++var)
            {
                const int n_dofs = num_dofs(var);
                for (int i = 0; i < n_dofs; ++i)
                {
                    for (int j = 0; j < n_dofs; ++j)
                    {
                        lm(dof_index(i, var), dof_index(j, var)) +=
                            wq * phi(i, q, var) * phi(j, q, var) * dJ;
                    }
                }
            }
        } // loop for q
    }
};


#endif