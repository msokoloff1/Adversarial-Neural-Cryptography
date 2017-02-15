##This file is used to perform analysis on the results from the adversarial network

#Custom function so that this can be run in any environment
loadPackage <- function(packageString) { 
	if(!require(packageString, character.only=TRUE)) {
		install.packages(packageString, repos="http://cran.rstudio.com/")
		if(!require(packageString,character.only = TRUE)) stop("Package not found")
	}
}

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  if (is.null(layout)) {
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  if (numPlots==1) {
    print(plots[[1]])
  } else {
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    for (i in 1:numPlots) {
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

#Load ggplot2
loadPackage('ggplot2')

files = list.files(path="../original/logs")

data = lapply(files, function(x){
  a = textConnection(readLines(paste0("../original/logs/",x))[-1])
  read.csv(a, header = TRUE, stringsAsFactors = FALSE)
})

require(data.table)
combinedData = rbindlist(data)
combinedData$RunNumber <- factor(combinedData$RunNumber)


# First plot
plotIter1 <- ggplot(combinedData, aes(x=Iteration, y=Alice.Bob.Incorrect, group = RunNumber,  colour = RunNumber )) +
    geom_line() +
    ggtitle("Alice and Bob Reconstruction Accuracy (Results using Abadi et al. architecture)")

plotIter2 <- ggplot(combinedData, aes(x=Iteration, y=EveIncorrect, group = RunNumber,  colour = RunNumber )) +
  geom_line() +
  ggtitle("Eve Reconstruction Accuracy (Results using Abadi et al. architecture)")



<<<<<<< HEAD
    
plot(plotIter1)
       
#multiplot(plotIter1, plotIter2, plotIter3, plotIter4, cols=2)
=======
multiplot(plotIter1, plotIter2,cols=1)
>>>>>>> 9dfb9a614f6b31a4b6f5d61cdcaef63920fc2e17

### Take the average over different length messages. Create box plots to compare.
###Do the same for different optimizers, etc. 

