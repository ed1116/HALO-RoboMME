import h5py

def recursive_print(group, indent=0):
    if isinstance(group, h5py.Group):
        for key in group.keys():
            print(f"{'  ' * indent}{key}")
            # print_attributes(group[key], indent+2)
            if isinstance(group[key], h5py.Group):
                recursive_print(group[key], indent+2)
            elif isinstance(group[key], h5py.Dataset):
                print(f"{'  ' * indent}{key}: {group[key].shape}")
    elif isinstance(group, h5py.Dataset):
        print(f"{'  ' * indent}{group.name}: {group.shape}")
    else:
        print(f"{'  ' * indent}{group.name}: {group.shape}")

def print_hdf5_structure(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        recursive_print(f["data"], 0)
    return True

# print all the attributes of the group
def print_attributes(group, indent=0):
    for key in group.attrs.keys():
        print(f"{'  ' * indent}{key}: {group.attrs[key]}")

def print_max_ep_length(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        max_ep_length = 0
        for key in f["data"].keys():
            if isinstance(f["data"][key], h5py.Group):
                max_ep_length = max(max_ep_length, len(f["data"][key]["states"]))
        print(f"Max episode length: {max_ep_length}")
    return max_ep_length

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", type=str, required=True)
    args = parser.parse_args()
    print_hdf5_structure(args.hdf5_path)
    print_max_ep_length(args.hdf5_path)
