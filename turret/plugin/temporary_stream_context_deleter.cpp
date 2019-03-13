#include "temporary_stream_context_deleter.hpp"

namespace turret {

TemporaryStreamContextDeleter::TemporaryStreamContextDeleter()
	: m_main_completion_event(nullptr)
	, m_plugin_completion_event(nullptr)
{ }

TemporaryStreamContextDeleter::TemporaryStreamContextDeleter(
	CUevent main_completion_event,
	CUevent plugin_completion_event)
	: m_main_completion_event(main_completion_event)
	, m_plugin_completion_event(plugin_completion_event)
{ }

TemporaryStreamContextDeleter::~TemporaryStreamContextDeleter(){
	if(m_main_completion_event){
		cuEventDestroy(m_main_completion_event);
	}
	if(m_plugin_completion_event){
		cuEventDestroy(m_plugin_completion_event);
	}
}


void release_temporary_stream_events(
	CUstream stream, CUresult status, void *data)
{
	auto ptr = reinterpret_cast<TemporaryStreamContextDeleter *>(data);
	delete ptr;
}

}
