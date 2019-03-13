#ifndef TURRET_ERROR_HPP
#define TURRET_ERROR_HPP

#include <stdexcept>
#include "Python.h"

namespace turret {

class PythonError : public std::runtime_error {

public:
	PythonError()
		: std::runtime_error("")
	{ }

};

void pyerror_test(){
	PyObject *err = PyErr_Occurred();
	if(err){ throw PythonError(); }
}

}

#endif
