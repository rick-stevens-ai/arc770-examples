#ifndef PRECISION_HPP
#define PRECISION_HPP

// Define the precision type for calculations
#ifdef USE_DOUBLE
    using PREC_TYPE = double;
#else
    using PREC_TYPE = float;
#endif

#endif // PRECISION_HPP
