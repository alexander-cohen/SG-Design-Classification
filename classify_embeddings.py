import dill

def load_points(n):
    fn = "saved_classification/all_unique_sg_{}.dill".format(n)
    with open(fn, "rb") as dataf:
        all_sols = dill.load(dataf)

    print("Found {} solutions for {} points".format(len(all_sols), n))
    for i,s in enumerate(all_sols):
        for l in s:
            print(l)
        print("\n\n")