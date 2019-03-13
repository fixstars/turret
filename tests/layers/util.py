# -*- coding: utf-8 -*-
import turret

def build_network(network_generator):
    builder = turret.InferenceEngineBuilder(
        turret.loggers.ConsoleLogger(turret.Severity.INFO))
    network = builder.create_network()
    network_generator(network)
    return network

def execute_inference(inputs, network_generator, max_batch_size=128):
    builder = turret.InferenceEngineBuilder(
        turret.loggers.ConsoleLogger(turret.Severity.INFO))
    network = builder.create_network()
    network_generator(network)
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = 1 << 30
    engine = builder.build(network)
    with turret.ExecutionContext(engine) as ctx:
        buffer = ctx.create_buffer()
        for k, v in inputs.items():
            buffer.put(k, v)
        ctx.execute(buffer)
        return buffer.get("output")
