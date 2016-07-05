import re

CYCLES_PER_SEC = 1.6 * 1000000000;

def create_report(runs, filename, param_info, arch_explore):
    # Create a report on the benchmarks.
    # This is not meant to be a generic report of information
    # but a very specific custom report that will change over time as we
    # feel different aspects of the benchmarks are important.

    def find_run(prog, param_spec, exact=False):
        for (params, logname, data) in runs[prog]:
            if exact and len(param_spec) != len(params):
                continue
            params = dict(params)
            satisfies_params = True
            for (k, v) in param_spec.items():
                if k in params:
                   if params[k] != v:
                       satisfies_params = False
                       break
                else:
                    if param_info[k]['default'] != v:
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

    def MB(n):
        return n / (1024*1024)

    def get_gflops(data):
        flops = data[0]['FLOPS']
        cycles = get_total_cycles(data[0])
        us_per_image = cycles / CYCLES_PER_SEC * 1000000
        return flops / us_per_image / 1000

    def get_tflops(data):
        return get_gflops(data)/1000

    alexnet_1_ipu = find_run('alexnet', {'--ipus': '1'})
    resnet34 = find_run('resnet34b', {'--graph-reuse': '1'})
    resnet34_no_reuse = find_run('resnet34b', {'--graph-reuse': '0'})
    resnet50 = find_run('resnet50', {'--graph-reuse': '1'})
    resnet50_no_reuse = find_run('resnet50', {'--graph-reuse': '0'})
    alexnet_large_tiles = find_run('alexnet', {'--tiles-per-ipu': '608'})
    alexnet_xlarge_tiles = find_run('alexnet', {'--tiles-per-ipu': '304'})
    alexnet_exchange8 = find_run('alexnet',
                                 {'--ipu-exchange-bandwidth':'8'}, True)
    alexnet_exchange8_reduce = find_run('alexnet',
                                           {'--ipu-exchange-bandwidth':'8',
                                            '--tiles-per-ipu':'1024'})
    alexnet_exchange16 = find_run('alexnet',
                                    {'--ipu-exchange-bandwidth':'16'}, True)
    alexnet_exchange16_reduce = find_run('alexnet',
                                           {'--ipu-exchange-bandwidth':'16',
                                            '--tiles-per-ipu':'1024'})

    resnet34_large_tiles = find_run('resnet34b', {'--tiles-per-ipu': '608'})
    resnet34_xlarge_tiles = find_run('resnet34b', {'--tiles-per-ipu': '304'})
    resnet34_exchange8 = find_run('resnet34b',
                                     {'--ipu-exchange-bandwidth':'8'}, True)
    resnet34_exchange8_reduce = find_run('resnet34b',
                                           {'--ipu-exchange-bandwidth':'8',
                                            '--tiles-per-ipu':'1024'})
    resnet34_exchange16 = find_run('resnet34b',
                                     {'--ipu-exchange-bandwidth':'16'}, True)
    resnet34_exchange16_reduce = find_run('resnet34b',
                                           {'--ipu-exchange-bandwidth':'16',
                                            '--tiles-per-ipu':'1024'})

    resnet50_large_tiles = find_run('resnet50', {'--tiles-per-ipu': '608'})
    resnet50_xlarge_tiles = find_run('resnet50', {'--tiles-per-ipu': '304'})
    resnet50_exchange8 = find_run('resnet50',
                                     {'--ipu-exchange-bandwidth':'8'}, True)
    resnet50_exchange8_reduce = find_run('resnet50',
                                           {'--ipu-exchange-bandwidth':'8',
                                            '--tiles-per-ipu':'1024'})
    resnet50_exchange16 = find_run('resnet50',
                                     {'--ipu-exchange-bandwidth':'16'}, True)
    resnet50_exchange16_reduce = find_run('resnet50',
                                           {'--ipu-exchange-bandwidth':'16',
                                            '--tiles-per-ipu':'1024'})

    ipu_tiles = 1216
    ipu_total_mem = ipu_tiles * 256 * 1024


    with open(filename, "w") as f:
        f.write('ALEXNET 1 IPU SUMMARY,\n,\n')
        alexnet_total_cycles = get_total_cycles(alexnet_1_ipu[0])
        alexnet_us_per_image = alexnet_total_cycles / CYCLES_PER_SEC * 1000000
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
            if len(layer_id_components) <= 2 and (len(layer_id_components) == 1 or layer_id_components[0][0] == 'F' or layer_id_components[1] == 'zero' or re.match('.*11x11.*',layer_id_components[0]) or re.match('.*Max.*', layer_id_components[0])):
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
        if resnet34_no_reuse:
            f.write('Resnet 34, {:.0f}, {:.0f}\n'.format(resnet34_no_reuse[0]['Number of vertices'],
                                                         resnet34_no_reuse[0]['Number of edges']))
        f.write('Resnet 34 (graph reuse), {:.0f}, {:.0f}\n'.format(resnet34[0]['Number of vertices'],
                                           resnet34[0]['Number of edges']))
        if resnet50_no_reuse:
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
        bytes_per_param = 2
        alexnet_params_mb = MB(alexnet_1_ipu[0]['Parameters'] * bytes_per_param)
        resnet34_params_mb = MB(resnet34[0]['Parameters'] * bytes_per_param)
        resnet50_params_mb = MB(resnet50[0]['Parameters'] * bytes_per_param)
        f.write('Parameters (MB), {:.0f},{:.0f},{:.0f}\n'.format(
                alexnet_params_mb,
                resnet34_params_mb,
                resnet50_params_mb))
        f.write('Tensor data/param, {:.2f},{:.2f},{:.2f}\n'.format(
                MB(alexnet_1_ipu[0]['Tensor data']) / alexnet_params_mb,
                MB(resnet34[0]['Tensor data']) / resnet34_params_mb,
                MB(resnet50[0]['Tensor data']) / resnet50_params_mb))
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


        f.write(',\nPERFORMANCE,\n,\n')

        alexnet_cycles = get_total_cycles(alexnet_1_ipu[0])
        alexnet_compute_ratio = \
          alexnet_1_ipu[0]['Compute cycles'] / alexnet_cycles
        resnet34_cycles = get_total_cycles(resnet34[0])
        resnet34_compute_ratio = \
          resnet34[0]['Compute cycles'] / resnet34_cycles
        resnet50_cycles = get_total_cycles(resnet50[0])
        resnet50_compute_ratio = \
          resnet50[0]['Compute cycles'] / resnet50_cycles

        alexnet_flops = alexnet_1_ipu[0]['FLOPS']
        resnet34_flops = resnet34[0]['FLOPS']
        resnet50_flops = resnet50[0]['FLOPS']
        alexnet_us_per_image = alexnet_cycles / CYCLES_PER_SEC * 1000000
        alexnet_gflops_per_sec = alexnet_flops / alexnet_us_per_image / 1000
        resnet34_us_per_image = resnet34_cycles / CYCLES_PER_SEC * 1000000
        resnet34_gflops_per_sec = resnet34_flops / resnet34_us_per_image / 1000
        resnet50_us_per_image = resnet50_cycles / CYCLES_PER_SEC * 1000000
        resnet50_gflops_per_sec = resnet50_flops / resnet50_us_per_image / 1000

        alexnet_vertex_ratio =  alexnet_1_ipu[0]['Perfect cycles'] / \
                                alexnet_1_ipu[0]['Compute cycles']
        resnet34_vertex_ratio = resnet34[0]['Perfect cycles'] / \
                                resnet34[0]['Compute cycles']
        resnet50_vertex_ratio = resnet50[0]['Perfect cycles'] / \
                                resnet50[0]['Compute cycles']

        alexnet_ratio = alexnet_1_ipu[0]['Perfect cycles'] / alexnet_cycles
        resnet34_ratio = resnet34[0]['Perfect cycles'] / resnet34_cycles
        resnet50_ratio = resnet50[0]['Perfect cycles'] / resnet50_cycles

        f.write(', Alexnet, ResNet34, ResNet50\n')
        f.write('Effective GFLOP/s,{:.0f},{:.0f},{:.0f}\n'.format(
                alexnet_gflops_per_sec,
                resnet34_gflops_per_sec,
                resnet50_gflops_per_sec))

        f.write(',,,\n')
        f.write('Compute ratio,{:.2f},{:.2f},{:.2f}\n'.format(
                alexnet_compute_ratio,
                resnet34_compute_ratio,
                resnet50_compute_ratio))
        f.write('Vertex overhead ratio,{:.2f},{:.2f},{:.2f}\n'.format(
                alexnet_vertex_ratio,
                resnet34_vertex_ratio,
                resnet50_vertex_ratio))
        f.write('Overall ratio,{:.2f},{:.2f},{:.2f}\n'.format(
                alexnet_ratio,
                resnet34_ratio,
                resnet50_ratio))

        f.write(',\nSUMMARY,\n,\n')
        f.write('Alexnet: {} cycles, {:.1f} MB\n'.format(alexnet_cycles, MB(get_total_mem(alexnet_1_ipu[0]))))
        f.write('Resnet34: {} cycles, {:.1f} MB\n'.format(resnet34_cycles, MB(get_total_mem(resnet34[0]))))
        f.write('Resnet50: {} cycles, {:.1f} MB\n'.format(resnet50_cycles, MB(get_total_mem(resnet50[0]))))

        def info(*runs):
            return [ get_tflops(runs[0]),
                     ((get_tflops(runs[0]) / get_tflops(alexnet_1_ipu)) - 1) * 100,
                     get_tflops(runs[1]),
                     ((get_tflops(runs[1]) / get_tflops(resnet34)) - 1) * 100,
                     get_tflops(runs[2]),
                     ((get_tflops(runs[2]) / get_tflops(resnet50)) - 1) * 100]

        if arch_explore:
            f.write(',\nARCHITECTURE EXPLORATION,\n,\n')
            f.write(',Alexnet,ResNet34,ResNet50\n')
            f.write('64Bit exchange, {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%)\n'.
                    format(*info(alexnet_exchange8,
                                 resnet34_exchange8,
                                 resnet50_exchange8)))
            f.write('64Bit exchange - reduced, {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%)\n'.
                    format(*info(alexnet_exchange8_reduce,
                                 resnet34_exchange8_reduce,
                                 resnet50_exchange8_reduce)))
            f.write('128Bit exchange, {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%)\n'.
                    format(*info(alexnet_exchange16,
                                 resnet34_exchange16,
                                 resnet50_exchange16)))
            f.write('128Bit exchange - reduced, {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%)\n'.
                    format(*info(alexnet_exchange16_reduce,
                                 resnet34_exchange16_reduce,
                                 resnet50_exchange16_reduce)))
            f.write('608 tiles, {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%)\n'.
                    format(*info(alexnet_large_tiles,
                                 resnet34_large_tiles,
                                 resnet50_large_tiles)))
            f.write('304 tiles, {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%)\n'.
                    format(*info(alexnet_xlarge_tiles,
                                 resnet34_xlarge_tiles,
                                 resnet50_xlarge_tiles)))

        def meminfo(*runs):
            return [ MB(get_total_mem(runs[0][0])),
                     ((get_total_mem(runs[0][0]) / get_total_mem(alexnet_1_ipu[0])) - 1) * 100,
                     MB(get_total_mem(runs[1][0])),
                     ((get_total_mem(runs[1][0]) / get_total_mem(resnet34[0])) - 1) * 100,
                     MB(get_total_mem(runs[2][0])),
                     ((get_total_mem(runs[2][0]) / get_total_mem(resnet50[0])) - 1) * 100 ]

        if arch_explore:
            f.write(',Alexnet,ResNet34,ResNet50\n')
            f.write('64Bit exchange, {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%)\n'.
                    format(*meminfo(alexnet_exchange8,
                                    resnet34_exchange8,
                                    resnet50_exchange8)))
            f.write('64Bit exchange - reduced, {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%)\n'.
                    format(*meminfo(alexnet_exchange8_reduce,
                                    resnet34_exchange8_reduce,
                                    resnet50_exchange8_reduce)))
            f.write('128Bit exchange, {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%)\n'.
                    format(*meminfo(alexnet_exchange16,
                                    resnet34_exchange16,
                                    resnet50_exchange16)))
            f.write('128Bit exchange - reduced, {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%)\n'.
                    format(*meminfo(alexnet_exchange16_reduce,
                                    resnet34_exchange16_reduce,
                                    resnet50_exchange16_reduce)))
            f.write('608 tiles, {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%)\n'.
                    format(*meminfo(alexnet_large_tiles,
                                    resnet34_large_tiles,
                                    resnet50_large_tiles)))
            f.write('304 tiles, {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%), {:.1f} ({:+.2f}%)\n'.
                    format(*meminfo(alexnet_xlarge_tiles,
                                    resnet34_xlarge_tiles,
                                    resnet50_xlarge_tiles)))
        
        def get_send_stats(data):
            min_send_ratio = 1
            max_send_ratio = 0
            sum_send_ratio = 0
            sum_cycles = 0
            for d in data:
                exchange_cycles = sum_fields(d, ['Send',
                                                 'Receive mux',
                                                 'Receive ptr',
                                                 'Nop'])
                if exchange_cycles == 0:
                    continue
                ratio = d['Send'] / exchange_cycles
                min_send_ratio = min(min_send_ratio, ratio)
                max_send_ratio = max(max_send_ratio, ratio)
                sum_send_ratio += ratio * exchange_cycles
                sum_cycles += exchange_cycles

            return (min_send_ratio * 100, sum_send_ratio / sum_cycles*100, max_send_ratio*100)

        def get_recv_stats(data):
            min_recv_ratio = 1
            max_recv_ratio = 0
            sum_recv_ratio = 0
            sum_cycles = 0
            for d in data:
                exchange_cycles = sum_fields(d, ['Send',
                                                 'Receive mux',
                                                 'Receive ptr',
                                                 'Nop'])
                if exchange_cycles == 0:
                    continue
                ratio = d['Exchange activity'] / 100
                min_recv_ratio = min(min_recv_ratio, ratio)
                max_recv_ratio = max(max_recv_ratio, ratio)
                sum_recv_ratio += ratio * exchange_cycles
                sum_cycles += exchange_cycles

            return (min_recv_ratio * 100, sum_recv_ratio / sum_cycles*100, max_recv_ratio*100)


        f.write(',\nEXCHANGE DENSITY,\n,\n')

        f.write('% cycles, min, mean, max\n')
        all_layers = alexnet_1_ipu[1:] + resnet34[1:] + resnet50[1:]
        f.write('Sending, {:.1f}%, {:.1f}%, {:.1f}%\n'.format(*get_send_stats(all_layers)))
        f.write('Receiving, {:.1f}%, {:.1f}%, {:.1f}%\n'.format(*get_recv_stats(all_layers)))

        f.write(',\nALEXNET TILE USAGE,\n,\n')

        f.write('Layer, #computing, #exchanging\n')
        computing_layers = [d for d in alexnet_1_ipu[1:-1] if d['Num tiles computing'] != 0]
        for i, d in enumerate(computing_layers) :
            f.write('{}, {}, {}\n'.format(d['Layer ID'],
                                          int(d['Num tiles computing']),
                                          int(d['Num tiles exchanging'])))

        f.write(',\nRESNET 50 TILE USAGE,\n,\n')

        f.write('Layer, #computing, #exchanging\n')
        computing_layers = [d for d in resnet50[1:-1] if d['Num tiles computing'] != 0]

        for i, d in enumerate(computing_layers) :
            f.write('{}, {}, {}\n'.format(d['Layer ID'],
                                          int(d['Num tiles computing']),
                                          int(d['Num tiles exchanging'])))
            
