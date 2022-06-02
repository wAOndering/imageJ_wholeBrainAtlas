import glob
import pandas as pd
import os
import warnings
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 
warnings.filterwarnings("ignore")

## TODO recreate the index from the excel file
## TODO matching of the files (improve and fail save either with sorting or character matching)

def paramsForCustomPlot(data, variableLabel='genotype', valueLabel='value', sort=True, **kwargs):
    """Function to create the parametters for ploting the variable, value, subject setup manually create a dictionary for the parameter to be reused for ploting see
    
    Parameters:
        data (DataFrame): dataframe with the data
        myPal (list): list of hexadecimal RGB value should be at least the length of the variableLable // this was removed due to change in default settings
        variableLabel (str): name of the variable of interest, header of the variable column
        subjectLabel (str): name of the subject of interest, header of the subject column
        valueLabel (str): name of the subject of interest, 
    """

    subjectLabel = kwargs.get('subjectLabel', None)
    if subjectLabel is None:
        subjectLabel = 'tmpSub'
        data.index = data.index.set_names(['tmpSub'])
        data = data.reset_index()
    dfSummary=data.groupby([variableLabel,subjectLabel]).mean()
    dfSummary.reset_index(inplace=True)
    if sort == True:
        dfSummary =  dfSummary.sort_values(by=[variableLabel],  ascending=False)
        data = data.sort_values(by=[variableLabel],  ascending=False)

    params = dict(  data=dfSummary,
                    x=str(variableLabel),
                    y=str(valueLabel),
                    hue=str(variableLabel),
                    )

    paramsNest = dict( data=data,
                    x=str(variableLabel),
                    y=str(valueLabel),
                    hue=str(variableLabel),
                    )

    ## calculate the number of observation 
    tmpObs = data[[variableLabel, valueLabel]]
    tmpObs = tmpObs.dropna()
    nobs = tmpObs.groupby(variableLabel).count()
    # nobs = list(itertools.chain.from_iterable(nobs.values))
    # nobs = [str(x) for x in nobs]
    nobs = nobs.reset_index()
    nobs = nobs.sort_values(by=[variableLabel],  ascending=False)
    nobs = list(nobs[valueLabel])
    nmax = tmpObs.max()[-1]*1.1

    return params, paramsNest, nobs, nmax

def customPlot(params, paramsNest, dirName='C:/Users/Windows/Desktop/MAINDATA_OUTPUT', figName = ' '):
    mpl.rcParams['axes.prop_cycle'] = cycler(color=['#66c2a5','#fc8d62','#8da0cb'])
    """Function to create save the plot to determine directior
    
    Parameters:
        params (dict): set of parameters for plotting main data
        paramsNest (dict): set of parameters for plotting main data (subNesting level)
        dirName(str): string to determine the directory
    """

    # create the frame for the figure
    os.makedirs(dirName,exist_ok=True)

    # the figure size correspond to the size of a plot in inches
    f, ax = plt.subplots(figsize=(7, 7))

    ## add if the study was longitudinal / repeated measures
    # castdf=pd.pivot_table(df, values='value', index=['subject'], columns=['genotype'])
    # for i in castdf.index:
    #     ax.plot(['wt','het'], castdf.loc[i,['wt','het']], linestyle='-', color = 'gray', alpha = .3)

    # fill the figure with appropiate seaborn plot
    # sns.boxplot(dodge = 10, width = 0.2, fliersize = 2, **params)
    sns.violinplot(dodge = 5, split = False, inner = 'quartile', width=0.6, cut=1, **paramsNest, zorder=0)
    sns.stripplot(jitter=0.08, dodge=True, size=4, linewidth=1, **paramsNest, zorder=1) #edgecolor='white'

    # control the figure parameter with matplotlib control
    # this order enable to have transparency of the distribution violin plot
    plt.setp(ax.collections, alpha=.2)

    sns.stripplot(jitter=0.08, dodge=True, edgecolor='white', size=8, linewidth=1, **params, zorder=2)
    # the point plot enable to plot the mean and the standard error 
    # to have the "sd" or 95 percent confidence interval 
    # for sem ci=68
    sns.pointplot(ci=68, scale=1.2, dodge= -0.1, errwidth=4, **params, zorder=4, color='grey')
    sns.pointplot(ci=95, dodge= -0.1, errwidth=2, **params, zorder=4, color='grey')
    # plot the median could be done with the commented line below however this would be redundant 
    # since the median is already ploted in the violin plot
    # sns.pointplot(ci=None, dodge= -0.2, markers='X',estimator=np.median, **params)


    # label plot legend and properties
    ax.legend_.remove()
    sns.despine() 

    ax.set_ylabel(params.get('y'), fontsize=30)
    ax.tick_params(axis="y", labelsize=20)
    # ax.set_ylim([-5,5])

    ax.set_xlabel(params.get('x'), fontsize=30)
    ax.tick_params(axis="x", labelsize=20, pad=10) # could also use: ax.xaxis.labelpad = 10 // plt.xlabel("", labelpad=10) // or change globally rcParams['xtick.major.pad']='8'
    plt.title(figName, fontsize=30)
    plt.tight_layout()

    ### to obligate viewing of the plot
    # plt.show(block=False)


    # property to export the plot 
    # best output and import into illustrator are svg 
    plt.savefig(dirName+os.sep+figName+".png")#,     plt.show(block=False)
    plt.close()

class combineFile():
	'''
	this class is to combine files for a given sample/subject
	if there are mutliple subfolders or subregion they will be incorporated as well
	this should be modified to be incorporated for the tiff of interest
	'''
	def __init__(self, filePath):
		self.filePath = filePath # get all the tiff file those should be the base ref 
		self.sID = filePath.split(os.sep)[-1].split('_')[0]
		self.plate = filePath.split(os.sep)[-1].split('_')[1].split('.')[0]
		self.identifier = self.sID+'_'+self.plate
		self.mainDir = os.path.dirname(filePath)
		self.measureFile = glob.glob(self.mainDir+os.sep+'*'+self.identifier+'*V2.csv',  recursive=True)[0] #get the list of measures
		self.regionFile = glob.glob(self.mainDir+os.sep+'*'+self.identifier+'*.txt',  recursive=True)[0] #get the list of particles and their principal regions
		self.summary = glob.glob(self.mainDir+os.sep+'*'+self.identifier+'*V2.xlsx',  recursive=True)[0]	#get the summary
		#### be careful not to have duplicate file for this 

	def flagNAN(self):
		tmp = pd.read_csv(self.measureFile)
		tmpNAN = tmp[np.isnan(tmp['IntDen'])]

		if len(tmpNAN) != 0: 
			alpha = pd.DataFrame({'file': [self.measureFile], 'rows (n)': [len(tmp)], 'rowsNAN (n)': [len(tmpNAN)]})
			return alpha
	

	def reIndexCreation(self):
		
		def aggCol(df, cols):
			''' Function to aggreegate columns 
			df: is a dataframe
			col: common name element that will be used to identify the column to be aggregated
			'''
			acro_l = [x for x in tmp.columns if cols in x]
			tmpName = cols
			df[acro_l] = df[acro_l].astype('str')
			df[tmpName] = df[acro_l].T.agg('~'.join)
			df.drop(acro_l, axis=1, inplace=True)

			return df

		## TODO chack the structure and how are the summary saved is summary is a list or not
		# if len (self.summary) > 1:
			# print('see comments to deal with this specific situation where mutliple analysis are saved per animal and plate')

		tmp = pd.read_excel(self.summary)
		## deal with individual files within excel folder espcially relevant when summary have been concatenated
		# sectionSummarized = tmp['File'].unique()

		### prevent abnormal assignment when there is multiple excel files per folders
		tmp = tmp[tmp['File'].str.contains(self.identifier)]


		##section to recreate the index list 
		## first for all non empty brain region we need to add 1 to the number of particle to expand by this number
		## given the method present 
		tmp['Particle Count'] = tmp['Particle Count'] + 1
		# tmp.loc[tmp['Particle Count'] == 0, 'Particle Count'] = tmp.loc[tmp['Particle Count'] == 0, 'Particle Count']+1
		for i in ['Acronym', 'Name']:
			tmp = aggCol(tmp, i)

		## next merge columns to facilitate the expansion of the table
		reps = tmp['Particle Count'].tolist()# create the list of repetition that should be performed 
		tmp = tmp.reset_index() # IMPORTANT to keep track of the original indexing
		tmp = tmp.loc[np.repeat(tmp.index.values, reps)]
		tmp = tmp.reset_index(drop=True) 

		return tmp

	def combineAreaMeasure(self):
		
		def splitCol(df, col):
			''' Function to 
			'''
			df = df.join(df[col].str.split('~', expand=True).add_prefix(col))

			return df


		# make it specific for the file of interest for the brain region of interst
		tmp = self.reIndexCreation()
		tmpMeas = pd.read_csv(self.measureFile)
		tmpArea = pd.read_csv(self.regionFile, delimiter='/', names=['c1','c2','c3'])
		test = pd.concat([tmp, tmpMeas, tmpArea], axis=1)

		# get the final file
		test['sID'] = self.sID
		test['Brain section'] = self.plate
		## sanitiy check to see if the index reconstruction is actually working
		# b = test[test['Particle Count']==1]

		for i in ['Acronym', 'Name']:
			test = splitCol(test, i)

		return test




mypath = r'Y:\Madalyn\Analysis'
files = glob.glob(mypath+'/*/**/*V2.csv',  recursive=True)
for i, j in enumerate(files):
	print(i,j)


# t = combineFile(files[21])
# tmp = t.flagNAN()
# # # t.identifier
# # t.combineAreaMeasure()



masterFile = []
ERROR = []
filesWithNan = []
for i in files:
	print(i)
	try:
		tmpDat = combineFile(i)
		tmp = tmpDat.combineAreaMeasure()
		masterFile.append(tmp)
		# tmpNaN = tmpDat.flagNAN()
		# filesWithNan.append(tmpNaN)
	except:
		print('ERROR: the file listed were not processed: ')
		print(i)
		ERROR.append(i)
# filesWithNan = pd.concat(filesWithNan)
# filesWithNan.to_csv(mypath+os.sep+'flagedNAN.csv')
masterFile = pd.concat(masterFile)

## create a unique index reference
masterFile = masterFile.reset_index()
masterFile = masterFile.rename(columns={'index':'index0', 'level_0':'index1'})
masterFile = masterFile.reset_index()

## read the group file
sID = pd.read_excel(r'Y:\Madalyn\Analysis\Groups.xlsx')
sID = sID.rename(columns={'Mouse':'sID'})

## merge the files
masterFile['sID'] = masterFile['sID'].astype('int')
masterFile = pd.merge(masterFile, sID, on='sID')

## need to drop the nan in c3 as the empty one correpsond to fake reference and not a cell
masterFile = masterFile.dropna(subset=['c3'])


## save the files
masterFile.to_csv(mypath+os.sep+'masterFile.csv')
pd.DataFrame({'err':ERROR}).to_csv(mypath+os.sep+'error.csv')

############################################
###### Filtering data out
############################################
toExclude = pd.read_csv(r"Y:\Madalyn\Analysis\toExclude.csv")
masterFile = pd.read_csv(r"Y:\Madalyn\Analysis\masterFile.csv")

for i,j in toExclude.iterrows():
	print(j)
	if type(j['side']) != str:
		masterFile = masterFile[~(masterFile['File'].str.contains(j['list']))]
	else:
		masterFile = masterFile[~(masterFile['File'].str.contains(j['list']) & masterFile['Group'].str.contains(str(j['side'])))]
masterFile.to_csv(mypath+os.sep+'masterFile_withExclusion.csv')


############################################
###### For graphing 
############################################
#### List of potential measure to itterate over
myMeasure = masterFile.columns
for i,j in enumerate(myMeasure):
	print(i,j)
myMeasure = myMeasure[12:48]
## get a subset dataset for given brain region
subSetBrainRegion = masterFile['Acronym6'].unique()[1:]
subSetBrainRegion = ['ACA', 'ACB', 'PL', 'ILA', 'ORB', 'AI']
for i in subSetBrainRegion:
	sebSet = masterFile[masterFile['Acronym6'] == i]
	for j in myMeasure:
		# try:
		print(j)
		myFig = i+'_'+j
		params, paramsNest, nobs, nmax = paramsForCustomPlot(data=sebSet, variableLabel='Memory reactivated', subjectLabel='sID', valueLabel= j)
		customPlot(params, paramsNest, dirName=r'Y:\Madalyn\Analysis\outputs_withExclusion', figName=myFig)
		plt.close('all')
		# except:
		# 	print('ERROR: the file listed were not processed: ')
		# 	print(i,j)



