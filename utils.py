import sys, math, copy, json

import numpy as np
import random as rnd

def hybrid_select(options, weights, rw_threshold):
    if rnd.random() > rw_threshold:
        # Roulette wheel selection
        assigned_constituency = rnd.choices(options,
                                            weights=weights,
                                            k=1)[0]
    else:
        # Greedy selection
        assigned_constituency = options[list(weights).index(np.max(weights))]

    return assigned_constituency

def get_country_constituency_probabilities(constituency_ids, probabilities, country, order_by_val=False, reverse=False):
    country_mask = [constituency_id.startswith(country[0].upper()) for constituency_id in constituency_ids]
    masked_constituency_ids = np.extract(country_mask, constituency_ids)
    masked_probabilities = np.extract(country_mask, probabilities)
    if order_by_val: return [(con, prob) for prob, con in sorted(zip(masked_probabilities, masked_constituency_ids), reverse=reverse)]
    else: return masked_constituency_ids, masked_probabilities
    
def reset_constituency_values(constituency, templates):
    new_constituency = copy.deepcopy(templates['constituencies'])
    for key in ['name', 'country', 'region', 'is_protected', 'wards']:
        new_constituency[key] = constituency[key]
    new_constituency['election_results']['vote_share'] = copy.deepcopy(templates[new_constituency['country']])
    return new_constituency

def gps_to_ecef(coors, alt=0):
    lat = coors[1]
    lon = coors[0]
    rad_lat = lat * (math.pi / 180.0)
    rad_lon = lon * (math.pi / 180.0)

    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / math.sqrt(1 - e2 * math.sin(rad_lat) * math.sin(rad_lat))

    x = (v + alt) * math.cos(rad_lat) * math.cos(rad_lon)
    y = (v + alt) * math.cos(rad_lat) * math.sin(rad_lon)
    z = (v * (1 - e2) + alt) * math.sin(rad_lat)

    return [x, y, z]

def get_bool_flag(flag, default_val):
    if flag in sys.argv: return not default_val
    else: return default_val
    
def get_string_argument(flag, default_val, name, prefix=None, suffix=None):
    if default_val == "none": default_val = None
    if flag in sys.argv:
        args_index = sys.argv.index(flag)
        if args_index == len(sys.argv) - 1 or sys.argv[args_index + 1].startswith('-'):
            print("Class REDISTRICTER ERROR: No '{0}' value given:\n  ↳ '{1}' flag must be followed by an string".format(name, flag))
            return None
        else:
            value = sys.argv[args_index + 1]
            if prefix != None and not value.startswith(prefix):
                print("Class REDISTRICTER ERROR: Invalid '{0}' value given:\n  ↳ '{1}' flag must begin with '{2}' and '{3}' is invalid.".format(name, flag, prefix, value))
                return None
            if suffix != None and not value.endswith(suffix):
                print("Class REDISTRICTER ERROR: Invalid '{0}' value given:\n  ↳ '{1}' flag must end with '{2}' and '{3}' is invalid.".format(name, flag, suffix, value))
                return None
            return value
    else: return default_val

def get_int_arguments(flag, default_val, name, val_range=None):
    if default_val == "none": default_val = None
    if flag in sys.argv:
        args_index = sys.argv.index(flag)
        if args_index == len(sys.argv) - 1 or sys.argv[args_index + 1].startswith('-'):
            print("Class REDISTRICTER ERROR: No '{0}' value given:\n  ↳ '{1}' flag must be followed by an int".format(name, flag))
            return None
        else:
            if sys.argv[args_index + 1].isdigit():
                value = int(sys.argv[args_index + 1])
                if val_range != None:
                    if (value >= val_range[0] and value <= val_range[1]): return value
                    else:
                        print("Class REDISTRICTER ERROR: Invalid '{0}' value given:\n  ↳ '{0}' must be 'int' in range ({1} <= x <= {2})".format(name, val_range[0], val_range[1]))
                        return None
                else: return value
            else:
                print("Class REDISTRICTER ERROR: Invalid '{0}' value given:\n  ↳ '{0}' must be an int, not '{1}'".format(name, sys.argv[args_index + 1]))
                return None
    else: return default_val

def get_float_arguments(flag, default_val, name, val_range=None):
    if default_val == "none": default_val = None
    if flag in sys.argv:
        args_index = sys.argv.index(flag)
        if args_index == len(sys.argv) - 1 or sys.argv[args_index + 1].startswith('-'):
            print("Class REDISTRICTER ERROR: No '{0}' value given:\n  ↳ '{1}' flag must be followed by a float ({2} <= x <= {3})".format(name, flag, val_range[0], val_range[1]))
            return None
        else:
            value = float(sys.argv[args_index + 1])
            if val_range != None:
                if (value >= val_range[0] and value <= val_range[1]): return value
                else:
                    print("Class REDISTRICTER ERROR: Invalid '{0}' value given:\n  ↳ '{0}' must be 'int' in range ({1} <= x <= {2})".format(name, val_range[0], val_range[1]))
                    return None
            else: return value
    else: return default_val

def print_manual(parameter_dict):
    left_margin = max([len(param['flag']) for param in parameter_dict.values()]) + 1
    cross_line = " +" + ("-"*(left_margin+2)) + "+"
    print(cross_line+"\n |"+(" "*(len(cross_line)-(len("param")+4)))+"param | UK Redistricter - Command Line Arguments\n"+cross_line)
    for key, param in parameter_dict.items():
        flag_margin = " " * (left_margin - len(param['flag']))
        margin = " " * left_margin
        flag_margin = " | " + flag_margin + param['flag'] + " | "
        margin = " | " + margin + " | "
        print(flag_margin + key + " [" +param['type'] + "]:")
        if 'name' in param.keys(): print(margin + "  Parameter:\n" + margin + "    " + param['name'])
        if 'desc' in param.keys():
            description = param['desc']
            if '\n' in description: description = description.replace("\n", "\n" + margin + "    ")
            print(margin + "  Description:\n" + margin + "    " + description)
        if 'range' in param.keys(): print(margin + "  Range: [" + str(param['range'][0]) + " - " + str(param['range'][1]) + "]")
        if 'prefix' in param.keys(): print(margin + "  Prefix: '" + param['prefix'] + "'")
        if 'suffix' in param.keys(): print(margin + "  Suffix: '" + param['suffix'] + "'")
        print(cross_line)
    print()
    exit()

def get_parameters(parameter_file):
    with open(parameter_file, 'r') as pfile:
        default_params = json.load(pfile)['default_params']

    if '-h' in sys.argv:
        print_manual(default_params)
        exit()
    else:
        parameters = {}
        for key, param_info in default_params.items():
            if 'range' in param_info.keys(): val_range = param_info['range']
            else: val_range = None
            
            if param_info['type'].upper() == 'INT':
                parameters[key] = get_int_arguments(param_info['flag'], param_info['val'], param_info['name'], val_range)
            elif param_info['type'].upper() =='FLOAT':
                parameters[key] = get_float_arguments(param_info['flag'], param_info['val'], param_info['name'], val_range)
            elif param_info['type'].upper() == 'BOOL':
                parameters[key] = get_bool_flag(param_info['flag'], param_info['val'])
            elif param_info['type'].upper() == 'STRING':
                if 'prefix' in param_info.keys(): prefix = param_info['prefix']
                else: prefix = None
                if 'suffix' in param_info.keys(): suffix = param_info['suffix']
                else: suffix = None
                parameters[key] = get_string_argument(param_info['flag'], param_info['val'], param_info['name'], prefix, suffix)

        for key, info in parameters.items():
            if info == None and default_params[key]['val'] != "none": exit()

        return parameters