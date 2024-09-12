class Utilities:
    
    def runtest(self, dat, flag):

        # plotit(dat, flag)

        # print('% Flagged : ', np.sum(flag) / (1.0 * np.prod(flag.shape)) * 100.0)

        return (np.sum(flag) / (1.0 * np.prod(flag.shape)) * 100.0), self.calcquality(dat, flag)


    def calcquality(self, dat, flag):
        """ Need to minimize the score that it returns"""

        shp = dat.shape

        npts = 0
        sumsq = 0.0
        maxval = 0.0
        leftover = []
        flagged = []
        for chan in range(0, shp[0]):
            for tm in range(0, shp[1]):
                val = np.abs(dat[chan, tm])
                if flag[chan, tm] == False:
                    leftover.append(val)
                else:
                    flagged.append(val)

        dmax, dmean, dstd = self.printstats(np.abs(dat[:, :]))
        rmax, rmean, rstd = self.printstats(leftover)
        fmax, fmean, fstd = self.printstats(flagged)

        maxdev = (rmax - rmean) / rstd
        fdiff = fmean - rmean
        sdiff = fstd - rstd

        # print("Max deviation after flagging : ", maxdev)
        # print("Diff in mean of flagged and unflagged : ", fdiff)
        # print("Std after flagging : ", rstd)
        
        ## Maximum deviation from the mean is 3 sigma. => Gaussian stats. 
        ## => What's leftover is noise-like and without significant outliers.
        aa = np.abs(np.abs(maxdev) - 3.0)

        ## Flagged data has a higher mean than what is left over => flagged only RFI. Maximize the difference between the means
        bb = 1.0 / ((np.abs(fdiff) - rstd) / rstd)
        
        ## Maximize the difference between the std of the flagged and leftover data => Assumes that RFI is widely varying...
        cc = 1.0 / (np.abs(sdiff) / rstd)

        ## Overflagging is bad
        dd = 0.0
        pflag = (len(flagged) / (1.0 * shp[0] * shp[1])) * 100.0
        if pflag > 70.0:
            dd = (pflag - 70.0)/10.0
        
        res = np.sqrt(aa ** 2 + bb ** 2 + cc * 2 + dd * 2)

        if (fdiff < 0.0):
            res = res + res + 10.0

        # print("Score : ", res)

        return res


    def printstats(self, arr):
        if (len(arr) == 0):
            return 0, 0, 1

        med = np.median(arr)
        std = np.std(arr)
        maxa = np.max(arr)
        mean = np.mean(arr)
        # print 'median : ', med
        # print 'std : ', std
        # print 'max : ', maxa
        # print 'mean : ', mean
        # print " (Max - mean)/std : ", ( maxa - mean ) / std

        return maxa, mean, std


    def getvals(self, col='DATA', vis="", spw="", scan=""):

        # print("SPW:", spw, "DDID:", ddid)

        self.tb.open(vis)
        if (spw and scan):
            self.tb.open(vis + '/DATA_DESCRIPTION')
            spwids = self.tb.getcol('SPECTRAL_WINDOW_ID')
            ddid = str(np.where(spwids == eval(spw))[0][0])
            tb1 = self.tb.query('SCAN_NUMBER==' + scan + ' && DATA_DESC_ID==' + ddid + ' && ANTENNA1=1 && ANTENNA2=2')
        else:
            tb1 = self.tb.query('ANTENNA1=1 && ANTENNA2=2')
        dat = tb1.getcol(col)
        tb1.close()
        self.tb.close()
        return dat


    def plotit(self, dat, flag):
        plt.clf()

        fig, ax = plt.subplots(2, 1, figsize=(10, 3),dpi=150)
        ax[0].imshow(np.abs(dat), vmin=0, vmax=100)
        ax[1].imshow(np.abs(dat * (1 - flag)), vmin=0, vmax=100)
