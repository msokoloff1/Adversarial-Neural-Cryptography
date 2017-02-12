##This file is used to perform analysis on the results from the adversarial network

#Custom function so that this can be run in any environment
loadPackage <- function(packageString) { 
	if(!require(packageString, character.only=TRUE)) {
		install.packages(packageString, repos="http://cran.rstudio.com/")
		if(!require(packageString,character.only = TRUE)) stop("Package not found")
	}
}

#Load ggplot2
loadPackage('ggplot2')


#Read in data
dataIter1 = readLines("../original/logs/log20170212_17_28_22.txt")[-1]
dataIter2 = readLines("../original/logs/log20170212_17_28_22.txt")[-1]
dataIter3 = readLines("../original/logs/log20170212_17_28_22.txt")[-1]
dataIter4 = readLines("../original/logs/log20170212_17_28_22.txt")[-1]


 
dataIter1 = read.csv(textConnection(dataIter1), header = TRUE, stringsAsFactors = FALSE)
dataIter2 = read.csv(textConnection(dataIter2), header = TRUE, stringsAsFactors = FALSE)
dataIter3 = read.csv(textConnection(dataIter3), header = TRUE, stringsAsFactors = FALSE)
dataIter4 = read.csv(textConnection(dataIter4), header = TRUE, stringsAsFactors = FALSE)



# First plot
plotIter1 <- ggplot(dataIter1, aes(x=iteration, y=abIncorrect)) +
    geom_line() +
    ggtitle("First iteration results")

    
       
#multiplot(plotIter1, plotIter2, plotIter3, plotIter4, cols=2)

### Take the average over different length messages. Create box plots to compare.
###Do the same for different optimizers, etc. 

