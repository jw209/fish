def divider():
    print("".center(80, '-'))

# function definition to display statistical summary of data
def statistical_summary(data):
    divider()
    for specie in data['Species'].unique():
        print(" {0} ".format(specie).center(80, '-'))
        divider()
        frame = data[data['Species'] == specie]
        print(" {0} median: ".format(specie).center(80, '-'))
        print(frame[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']].median())
        divider()
        print(" {0} mode: ".format(specie).center(80, '-'))
        print(frame[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']].mode().head(1))
        divider()
        print(frame.describe())
        divider()
