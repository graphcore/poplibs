def create_report(runs, filename):
    # Create a report on the benchmarks.
    # This is not meant to be a generic report of information
    # but a very specific custom report that will change over time as we
    # feel different aspects of the benchmarks are important.

    def find_run(prog, param_spec):
        for (run_prog, params, data) in runs:
            if run_prog != prog:
                continue
            satisfies_params = True
            for (k, v) in param_spec.items():
                if k not in params or params[k] != v:
                    satisfies_params = False
                    break
            if satisfies_params:
                return data

    def sum_fields(data, fields):
        return sum(data[k] for k in fields)

    def sum_fields_if_there(data, fields):
        return sum(data[k] for k in fields if k in data)

    def get_total_mem(data):
        return sum_fields(data, ['Vertex data',
                                 'Tensor data',
                                 'Pipelined output copies',
                                 'In edge pointers',
                                 'Message memory',
                                 'Run instructions',
                                 'Exchange supervisor code'])

    def get_total_cycles(data):
        return sum_fields_if_there(data, ['Compute cycles',
                                          'Global exchange cycles',
                                          'Send',
                                          'Receive mux',
                                          'Receive ptr',
                                          'Nop',
                                          'Tile sync',
                                          'IPU sync',
                                          'Global sync'])

    def MB(num_bytes):
        return num_bytes/(1024*1024)


    alexnet_1_ipu = find_run('alexnet', {'Num IPUs': 1})
    resnet34 = find_run('resnet34b', {'Reuse graphs': 1})
    resnet34_no_reuse = find_run('resnet34b', {'Reuse graphs': 0})
    resnet50 = find_run('resnet50', {'Reuse graphs': 1})
    resnet50_no_reuse = find_run('resnet50', {'Reuse graphs': 0})

    ipu_tiles = 1216
    ipu_total_mem = ipu_tiles * 256 * 1024

    cycles_per_sec = 1.6 * 1000000000;

    with open(filename, "w") as f:
        f.write('ALEXNET 1 IPU SUMMARY,\n,\n')
        alexnet_total_cycles = get_total_cycles(alexnet_1_ipu[0])
        alexnet_us_per_image = alexnet_total_cycles / cycles_per_sec * 1000000
        f.write('Alexnet 1 IPU time per image, {:.1f}us\n'.format(
                alexnet_us_per_image))

        alexnet_total_mem = get_total_mem(alexnet_1_ipu[0])
        alexnet_mem_percent = alexnet_total_mem / ipu_total_mem * 100
        f.write('Alexnet 1 IPU total mem, {:.1f}%\n'.format(alexnet_mem_percent))

        f.write(',\nALEXNET 1 IPU LAYER BY LAYER CYCLE BREAKDOWN,\n,\n')

        layer_name = ''
        layer_total = 0
        layer_info = []
        for layer in alexnet_1_ipu[1:]:
            layer_id_components = layer['Layer ID'].split('.')
            this_layer_name = layer_id_components[0]
            if len(layer_id_components) <= 2:
                if layer_name:
                    layer_info.append((layer_name, layer_total))
                layer_name = this_layer_name
                layer_total = 0
            layer_total += get_total_cycles(layer)

        all_layer_total = sum([x[1] for x in layer_info])

        for (layer_name, layer_total) in layer_info:
            percent = layer_total / all_layer_total * 100
            f.write('{}, {:.1f}%\n'.format(layer_name, percent))

        f.write(',\nALEXNET 1 IPU CYCLE BREAKDOWN,\n,\n')

        all_cycles = get_total_cycles(alexnet_1_ipu[0])
        compute_cycles_percent = \
          alexnet_1_ipu[0]['Compute cycles'] / all_cycles * 100
        exchange_cycles_percent = \
          sum_fields(alexnet_1_ipu[0], ['Send',
                                       'Receive mux',
                                       'Receive ptr',
                                       'Nop']) / all_cycles * 100
        sync_cycles_percent = \
          sum_fields(alexnet_1_ipu[0], ['Tile sync',
                                       'IPU sync']) / all_cycles * 100
        f.write('Compute, {:.1f}%\n'.format(compute_cycles_percent))
        f.write('Exchange, {:.1f}%\n'.format(exchange_cycles_percent))
        f.write('Sync, {:.1f}%\n'.format(sync_cycles_percent))

        f.write(',\nALEXNET 1 IPU MEMORY BREAKDOWN,\n,\n')

        for field in ['Vertex data',
                      'Tensor data',
                      'In edge pointers',
                      'Message memory',
                      'Run instructions',
                      'Exchange supervisor code']:
            mem = alexnet_1_ipu[0][field]
            f.write('{}, {:.1f}%\n'.format(field, mem / alexnet_total_mem * 100))

        f.write(',\nALEXNET 1 IPU EXCHANGE BREAKDOWN,\n,\n')

        exchange_cycles = sum_fields(alexnet_1_ipu[0], ['Send',
                                                       'Receive mux',
                                                       'Receive ptr',
                                                       'Nop'])

        for field in ['Send', 'Receive mux', 'Receive ptr', 'Nop']:
            cycles_percent = alexnet_1_ipu[0][field] / exchange_cycles * 100
            f.write('{}, {:.1f}%\n'.format(field, cycles_percent))

        f.write(',\n')

        exchange_activity = alexnet_1_ipu[0]['Exchange activity']
        f.write('Receiving, {:.1f}%\n'.format(exchange_activity));
        f.write('Not receiving, {:.1f}%\n'.format(100-exchange_activity));

        f.write(',\nALEXNET 1 IPU PERFORMANCE,\n,\n')

        flops = alexnet_1_ipu[0]['FLOPS']
        gflops_per_sec = flops / alexnet_us_per_image / 1000
        f.write('Effective GFLOP/s,{:.0f}\n'.format(gflops_per_sec))
        vertex_ratio =  alexnet_1_ipu[0]['Perfect cycles'] / \
                         alexnet_1_ipu[0]['Compute cycles']
        f.write('Compute ratio, {:.2f}\n'.format(compute_cycles_percent/100))
        f.write('Vertex overhead ratio,{:.2f}\n'.format(vertex_ratio))
        overall_ratio = alexnet_1_ipu[0]['Perfect cycles'] / all_cycles
        f.write('Overall compute ratio,{:.2f}\n'.format(overall_ratio))

        f.write(',\nGRAPH SIZES,\n,\n')

        f.write(',Vertices,Edges\n')
        f.write('Alexnet, {:.0f}, {:.0f}\n'.format(alexnet_1_ipu[0]['Number of vertices'],
                                           alexnet_1_ipu[0]['Number of edges']))
        f.write('Resnet 34, {:.0f}, {:.0f}\n'.format(resnet34_no_reuse[0]['Number of vertices'],
                                           resnet34_no_reuse[0]['Number of edges']))
        f.write('Resnet 34 (graph reuse), {:.0f}, {:.0f}\n'.format(resnet34[0]['Number of vertices'],
                                           resnet34[0]['Number of edges']))
        f.write('Resnet 50, {:.0f}, {:.0f}\n'.format(resnet50_no_reuse[0]['Number of vertices'],
                                           resnet50_no_reuse[0]['Number of edges']))
        f.write('Resnet 50 (graph reuse), {:.0f}, {:.0f}\n'.format(resnet50[0]['Number of vertices'],
                                           resnet50[0]['Number of edges']))

        f.write(',\nMEMORY USAGE,\n,\n')

        f.write('Category, Alexnet, ResNet34, ResNet50\n')
        for field in ['Vertex data',
                      'Tensor data',
                      'In edge pointers',
                      'Message memory',
                      'Run instructions',
                      'Exchange supervisor code']:
            f.write('{} (MB), {:.0f},{:.0f},{:.0f}\n'.format(
                    field,
                    MB(alexnet_1_ipu[0][field]),
                    MB(resnet34[0][field]),
                    MB(resnet50[0][field])))
        f.write('TOTAL (MB), {:.0f},{:.0f},{:.0f}\n'.format(
                MB(get_total_mem(alexnet_1_ipu[0])),
                MB(get_total_mem(resnet34[0])),
                MB(get_total_mem(resnet50[0]))))
        f.write(',,,\n')
        f.write('Parameters (MB),,,\n')
        f.write('Tensor data/param (MB),,\n')
        f.write('Num vertices,{:.0f},{:.0f},{:.0f}\n'.format(
                alexnet_1_ipu[0]['Number of vertices'],
                resnet34[0]['Number of vertices'],
                resnet50[0]['Number of vertices'],
               ))
        f.write('Num edges,{:.0f},{:.0f},{:.0f}\n'.format(
                alexnet_1_ipu[0]['Number of edges'],
                resnet34[0]['Number of edges'],
                resnet50[0]['Number of edges'],
               ))
        vertex_bytes_fields = ['Vertex data', 'Run instructions']
        alexnet_vertex_bytes = sum_fields(alexnet_1_ipu[0], vertex_bytes_fields)
        resnet34_vertex_bytes = sum_fields(resnet34[0], vertex_bytes_fields)
        resnet50_vertex_bytes = sum_fields(resnet50[0], vertex_bytes_fields)
        f.write('Bytes/vertex,{:.1f},{:.1f},{:.1f}\n'.format(
                alexnet_vertex_bytes/alexnet_1_ipu[0]['Number of vertices'],
                resnet34_vertex_bytes/resnet34[0]['Number of vertices'],
                resnet50_vertex_bytes/resnet50[0]['Number of vertices'],
               ))
        edge_bytes_fields = ['In edge pointers', 'Exchange supervisor code']
        alexnet_edge_bytes = sum_fields(alexnet_1_ipu[0], edge_bytes_fields)
        resnet34_edge_bytes = sum_fields(resnet34[0], edge_bytes_fields)
        resnet50_edge_bytes = sum_fields(resnet50[0], edge_bytes_fields)
        f.write('Bytes/edge,{:.1f},{:.1f},{:.1f}\n'.format(
                alexnet_edge_bytes/alexnet_1_ipu[0]['Number of edges'],
                resnet34_edge_bytes/resnet34[0]['Number of edges'],
                resnet50_edge_bytes/resnet50[0]['Number of edges'],
               ))
