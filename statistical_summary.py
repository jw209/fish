def divider():
    print("".center(80, '-'))

# function definition to display statistical summary of data
def statistical_summary(data):
    divider()
    for specie in data['Species'].unique():
        print(" {0} ".format(specie).center(80, '-'))
        divider()
        frame = data[data['Species'] == specie]
        print(frame.describe())
        divider()
