#ifndef MY_INITIAL_CONDITION_H
#define MY_INITIAL_CONDITION_H

class InitialConditionAssembler : private AssemblyAssistant<DIM, double>
{

public:
   InitialConditionAssembler() {}

   void set_parameters(DataType x_location, DataType y_location, DataType variance, bool normed)
   {
       x_location_ = x_location;
       y_location_ = y_location;
       // mean_ = mean;
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
                       wq * exp( - ((x(q)[0] - x_location_) * (x(q)[0] - x_location_) / variance_) -
                            ((x(q)[1] - y_location_) * (x(q)[1] - y_location_) / variance_)) / std::sqrt(2*M_PI) *
                       dJ;
               }
           } // var
       }     // loop for q
   }

private:
   DataType x_location_, y_location_;
   DataType variance_; // mean_
   bool normed_;
};



double haversine_step_ellipse(double a, double x_q0, double center_x, double x_q1, double center_y, double r_x, double r_y, double transition_width) {
    double distance = a * (((x_q0 - center_x) / r_x) * ((x_q0 - center_x) / r_x) + ((x_q1 - center_y) / r_y) * ((x_q1 - center_y) / r_y));
    double transition = 1 / (1 + std::exp(4 * (distance - 1) / transition_width));
    return transition;
}

double haversine_step(double a, double x_q0, double center_x, double x_q1, double center_y, double r, double transition_width) {
    double distance = a * ((x_q0 - center_x) * (x_q0 - center_x) + (x_q1 - center_y) * (x_q1 - center_y));
    double transition = 0.5 / (1.0 + std::exp(4.0 * (distance - r * r) / (transition_width * transition_width)));
    return transition;
}


// class InitialConditionAssembler : private AssemblyAssistant<DIM, double>
// {
//
// public:
//     InitialConditionAssembler() {}
//
//    void set_parameters(DataType x_location, DataType y_location, DataType variance, bool normed)
//    {
//        x_location_ = x_location;
//        y_location_ = y_location;
//        // mean_ = mean;
//        variance_ = variance;
//        normed_ = normed;
//    }
//
//     void operator()(const Element<double, DIM> &element,
//                     const Quadrature<double> &quadrature, LocalVector &lv)
//     {
//         AssemblyAssistant<DIM, double>::initialize_for_element(element, quadrature,
//                                                                false);
//
//         const int num_q = num_quadrature_points();
//         DataType mat_num_ = element.get_cell().get_material_number();
//         double r_x = 0.15; //0.0009; // semi major axis in x
//         double r_y = 0.1; //semi major axis in y
//         double a_ellipse = 50.0;
//         double a = 30;
//         double r = 0.2; // Radius of the step
//         double transition_width = 0.002; // Adjust the transition width
//         double transition_width_ellipse = 0.8; // adjust for ellipse.
//
//         for (int q = 0; q < num_q; ++q)
//         {
//             const double wq = w(q);
//             const double dJ = std::abs(detJ(q));
//             const size_t num_dof = this->num_dofs_total();
//
//             for (int i = 0; i < num_dof; ++i)
//             {
//                 for (int var = 0; var < 1; ++var)
//                 {
//                     // if ((std::sin(7.2 * (x(q)[0]  - x_location_) + 5.6 * (x(q)[1] - y_location_) + 1) * std::pow(4 * (x(q)[0]  - x_location_) - 0.2, 2) + (std::sin(8 * (x(q)[0]  - x_location_)) + 1) * std::pow(64 * (x(q)[1] - y_location_), 2)) <= 0.09) {
//                     // if (((x(q)[0] - x_location_)*(x(q)[0] - x_location_)) + ((x(q)[1] - y_location_) * (x(q)[1] - y_location_)) <= r_init) {
//                     //     lv[i] += wq * 1.0 * dJ;
//                     // }else{
//                     //     lv[i] += wq * 0 * dJ;
//                     // }
//
//                     // double haversine_value = haversine_step(a, x(q)[0], x_location_, x(q)[1], y_location_, r, transition_width); // haversine step circle
//
//                     double haversine_value = haversine_step_ellipse(a_ellipse, x(q)[0], x_location_, x(q)[1], y_location_, r_x, r_y, transition_width_ellipse); // haversine step ellipse
//                     lv[i] += wq * haversine_value * dJ;
//
//                 }
//             } // var
//         }     // loop for q
//     }
//
// private:
//    DataType x_location_, y_location_;
//    DataType variance_; // mean_
//    bool normed_;
// };




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
