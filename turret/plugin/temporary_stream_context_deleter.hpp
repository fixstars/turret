#ifndef TURRET_PLUGIN_TEMPORARY_STREAM_CONTEXT_WRAPPER_HPP
#define TURRET_PLUGIN_TEMPORARY_STREAM_CONTEXT_WRAPPER_HPP

#include "cuda.h"
#include "Python.h"

namespace turret {

class TemporaryStreamContextDeleter {

private:
	CUevent m_main_completion_event;
	CUevent m_plugin_completion_event;

	TemporaryStreamContextDeleter(
		const TemporaryStreamContextDeleter&) = delete;
	TemporaryStreamContextDeleter& operator=(
		const TemporaryStreamContextDeleter&) = delete;

public:
	TemporaryStreamContextDeleter();
	TemporaryStreamContextDeleter(
		CUevent main_completion_event,
		CUevent plugin_completion_event);
	~TemporaryStreamContextDeleter();

};


void release_temporary_stream_events(
	CUstream stream, CUresult status, void *data);

}

#endif
