#include <nlopt.hpp>
#include <iostream>
#include <chrono>

/*
  x in R^2 x = [x0, x1]

  min sqrt(x1)
  s.t.
  -x1 <= 0
  -x1 + (-x0 + 1)^3 <= 0
  -x1 + (2x0)^3 <= 0

  ans: 0.54433 (x0, x1) = (1/3, 8/27)

*/

int count = 0;

double sqpCost(const std::vector<double> &x, std::vector<double> &grad, void *ptr)
{
  count++;
  if(grad.size() > 0)
    {
      grad.at(0) = 0;
      grad.at(1) = 0.5 / std::sqrt(x.at(1));
    }
  return std::sqrt(x.at(1));
}

void sqpConstraint(unsigned m, double *result, unsigned n, const double* x, double* grad, void* ptr)
{
  result[0] = -x[1];
  result[1] = -x[1] + (-x[0] + 1) * (-x[0] + 1) * (-x[0] + 1);
  result[2] = -x[1] + (2 * x[0]) * (2 * x[0]) * (2 * x[0]);

  if(grad == NULL) return;

  grad[0 * n + 0] = 0;
  grad[0 * n + 1] = -1 * 3 * (-x[0] + 1) * (-x[0] + 1);
  grad[0 * n + 2] = 2 * 3 * (2 * x[0]) * (2 * x[0]);
  grad[1 * n + 0] = -1;
  grad[1 * n + 1] = -1;
  grad[1 * n + 2] = -1;
}

int main()
{
  nlopt::opt solver(nlopt::LN_COBYLA, 2);
  solver.set_min_objective(sqpCost, NULL);
  solver.add_inequality_mconstraint(sqpConstraint, NULL, {1e-4, 1e-4, 1e-4});

  std::vector<double> lb(2);
  std::vector<double> ub(2);
  lb.at(0) = -INFINITY;
  lb.at(0) = 0;
  ub.at(0) = INFINITY;
  ub.at(1) = INFINITY;

  solver.set_lower_bounds(lb);
  solver.set_upper_bounds(ub);
  solver.set_xtol_rel(1e-6);
  solver.set_ftol_rel(1e-6);
  solver.set_maxeval(1000);

  std::vector<double> opt_x(2);
  opt_x.at(0) = 1.234;
  opt_x.at(1) = 5.678;

  double min_val;
  nlopt::result result;
  double duration;
  try
    {
      auto start = std::chrono::high_resolution_clock::now();
      result = solver.optimize(opt_x, min_val);
      auto end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
  catch(std::runtime_error error)
    {}
  if(result < 0) std::cout << "[nlopt] failed to solve. result is " << result << std::endl;
  else
    {
      std::cout << "solved with " << count << " iteration" << std::endl;
      std::cout << "final cost: " << min_val << std::endl;
      std::cout << "final solution: ";
      for(size_t i = 0; i < opt_x.size(); i++)
        {
          std::cout << opt_x.at(i) << " ";
        }
      std::cout << std::endl;
      std::cout << "solve time: " << duration << "[us]"<< std::endl;

    }


}
