from utils.optimisation import execute_optimisation
import argparse
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Execute optimisation algorithm.')
parser.add_argument('--server',
                    type=str,
                    help='Name of the server hosting the database')
parser.add_argument('--schema',
                    type=str,
                    help='Name of the schema to write to')
parser.add_argument('--name',
                    type=str,
                    help='Name of the resulting table in the database')
parser.add_argument('--capacity_upper_bound',
                    type=float,
                    help='Maximum percentage of resource capacity that can be planned in',
                    default=1,
                    required=False)
parser.add_argument('--weight_exponent',
                    type=float,
                    help='How heavily the priority of a project is weighed',
                    default=2,
                    required=False)
parser.add_argument('--resource_ids',
                    type=list,
                    help='Resources for which the planning is created. Use "None" for all resources simultaneously',
                    default=None,
                    required=False)
parser.add_argument('--shift_bounds',
                    type=tuple,
                    help='''
                    A tuple (x, y) of ints. x is the maximum number of months a project can be shifted backwards.
                    y is the maximum number of months a project can be shifted forwards.
                    ''',
                    default=(-float('inf'), float('inf')),
                    required=False)
parser.add_argument('--timeslot_capacity_multiplier',
                    type=dict,
                    help='''
                    Dictionary of the form {1: x1, 2: x2, ..., T: xT} where xt is the anticipated multiplier for a 
                    resource's capacity in timeslot t.
                    ''',
                    default={1: 1,
                             2: 1,
                             3: 1,
                             4: 1,
                             5: 1,
                             6: 1,
                             7: 1,
                             8: 1,
                             9: 1,
                             10: 1,
                             11: 1,
                             12: 1},
                    required=False)
args = parser.parse_args()


def main():
    name = args.name
    schema = args.schema
    server = args.server
    timeslot_capacity_multiplier = args.timeslot_capacity_multiplier
    capacity_upper_bound = args.capacity_upper_bound
    weight_exponent = args.weight_exponent
    shift_bounds = args.shift_bounds
    resource_ids = args.resource_ids
    selection, ongoing_projects, all_projects = execute_optimisation(
        server=server,
        timeslot_capacity_multiplier=timeslot_capacity_multiplier,
        capacity_upper_bound=capacity_upper_bound,
        weight_exponent=weight_exponent,
        shift_bounds=shift_bounds,
        resource_ids=resource_ids)
    print(selection)
    # Process the selected planning by e.g. uploading into SQL database, saving to Excel, etc.


if __name__ == '__main__':
    main()
