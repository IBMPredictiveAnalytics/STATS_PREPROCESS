import spss, spssaux, spssdata
from extension import Template, Syntax, processcmd
from spssdata import Spssdata, vdef 
from sklearn import preprocessing
import numpy as np
import random, pickle, time

# debugging
#try:
    #import wingdbstub
    #import threading
    #wingdbstub.Ensure()
    #wingdbstub.debugger.SetDebugThreads({threading.get_ident(): 1})
#except:
    #pass

# temporary
global _
try:
    _("---")
except:
    def _(msg):
        return msg

TFdict = {True: _("Yes"), False: _("No")}

# do STATS PREPROCESS command

def doprep(varnames=None, idvar=None, dataset=None, showstats=True, suffix=None, update=False, sorting=True,
        outtransfile=None, intransfile=None, 
        dostd=False, standardmean=True, standardstd=True,
        dorange=False, rangemin=None, rangemax=None, rangemaxabs=False, 
        dorobust=False, robustcenter=True, robustscale=False, robustqrange=[25., 75.],
        dononlinear=False, nonlinearmethod="yeojohnson", nonlinearstandardize=False,
        doquantile=False, quantilenquantiles=1000, quantiledist="uniform",
        dokbins=False, kbinsnbins=5, kbinsbindef="quantile"
    ):

    if isinstance(idvar, list):
        idvar = idvar[0]
    if isinstance(dataset, list):
        dataset = dataset[0]
    if isinstance(suffix, list):
        suffix = suffix[0]
    # CDB may generate ".pkl" for an empty browse control
    if intransfile is not None and intransfile.startswith("."):
        intransfile = None
    if outtransfile is not None and outtransfile.startswith("."):
        outtransfile = None
        
    # Validation

    dsactivename = spss.ActiveDataset()

    if spss.GetWeightVar() is not None:
        print(_("Warning: this procedure does not support weights."))
    if spss.GetSplitVariableNames():
        print(_("Warning: this procedure does not support split file settings."))
    if dsactivename == dataset:
        raise ValueError(_("The output dataset name must be different from the input name"))
    tasks = any([dostd, dorange, dorobust, dononlinear, doquantile, dokbins])
    if not any([tasks, intransfile]):
        raise ValueError(_("No transformations were requested."))  
    if tasks and intransfile:
        print(_("""Warning: When using an APPLYTRANS file, any other transform tasks are ignored."""))
    if dataset is None and not update:
        raise ValueError(_(f"""No new dataset is specified, and update is NO,
so no results would be produced."""))
    
    if intransfile:
        with open(intransfile, "rb") as f:
            translist = pickle.Unpickler(f).load()
        idvar = translist[1]
    else:
        translist = None
    # idvar could come from syntax or the intransfile, which would override
    if idvar is None:
        raise ValueError(_("An ID variable must be specified."))
    
    if dsactivename == "*":
        # assign a temporary name if necessary and remove it at the end
        if not update:
            raise ValueError(
                _("""The active dataset must have a dataset name if creating a new output dataset"""))
        dsactivename = "D" + str(random.uniform(.05, 1))
        spss.Submit(f"""DATASET NAME {dsactivename}""")
        tempdsactive = True
    else:
        tempdsactive = False
        
    if sorting:
        spss.Submit(f"""SORT CASES BY {idvar}.""")
        
    vardict = spssaux.VariableDict(caseless=True)
    if varnames is not None:
        varnames = vardict.expand(varnames)
        # case correct variable and id names
        varnames = [vardict[v].VariableName for v in varnames]
        if any(vardict[name].VariableType > 0 for name in varnames):
            raise ValueError(_("String variables cannot be used in this procedure"))
    idvar = vardict[idvar].VariableName  # case correcting
    
    if varnames is not None and intransfile:
        if [v.lower() for v in translist[2]] != [v.lower() for v in varnames]:
            raise ValueError(_("Variable names specified do not match names in saved transform file"))    
    if varnames is None and intransfile:
        varnames = translist[2]
    datasetnamed = dataset is not None
    if not datasetnamed:
        dataset = "D" + str(random.uniform(.05, 1.))
    
    # get data will all missing values set to None
    # Need a better way to do this retrieval
    # dtaid will be used to enable matching results to input dataset
    # The idvar must be separated off
    if varnames is None:
        raise ValueError(_("A variable names list and id variable must be specified"))
    dta = spssdata.Spssdata(indexes=[idvar] + varnames, names=False, convertUserMissing=True).fetchall()
    dtaid = [row[0] for row in dta]
    dta = np.array([row[1:] for row in dta])
    
    if translist == None:
        translist = [time.asctime(), idvar, varnames]
        
    # tasks
    spss.StartProcedure("Preprocess Data")
    if intransfile is None:
        try:
            if dostd:
                dta = dostdf(dta, varnames, standardmean, standardstd, showstats, vardict, translist)
            if dorange:
                dta = dorangef(dta, varnames, rangemin, rangemax, showstats, vardict, translist)
            if dononlinear:
                dta = dononlinearf(dta, varnames, nonlinearmethod, nonlinearstandardize, showstats, vardict, translist)
            if dorobust:
                dta = robustf(dta, varnames, robustcenter, robustscale, robustqrange, showstats, vardict, translist)
            if doquantile:
                dta = quantilef(dta, varnames, quantilenquantiles, quantiledist, showstats, vardict, translist)
            if dokbins:
                dta = kbinsf(dta, varnames, kbinsnbins, kbinsbindef, showstats, vardict, translist)
        except:
            spss.EndProcedure()
            raise
    else:
        try:
            dta = applytrans(dta, intransfile, varnames, showstats, translist)
        except:
            spss.EndProcedure()
            raise 
        
    # create new dataset with idvar and transformed variables
    createnewdataset(idvar, varnames, dtaid, dta, dataset, datasetnamed, dsactivename,
        update, suffix, vardict)
    spss.EndProcedure()
    spss.Submit(f"""DATASET ACTIVATE {dsactivename}.""")
    # If temporary dataset name was assigned to the active file, remove it (which leaves ds open)
    if tempdsactive:
        spss.Submit(f"""DATASET CLOSE *.""")
    if outtransfile:
        with open(outtransfile, "wb") as f:
            pickle.Pickler(f).dump(translist)
        print(_(f"Transform specifications written to file {outtransfile}, {translist[0]}"))
    
def dostdf(dta, varnames, standardmean, standardstd, showstats, vardict, translist):
    """Standardize data and return data matrix
    
    standardmean indicates whether to remove means
    standardstd indicates whether to remove standard deviation
    showstats indicates whether to display pivot table of statistics"""
    
    scaler = preprocessing.StandardScaler(with_mean=standardmean, with_std=standardstd).fit(dta)
    dta = scaler.fit_transform(dta)
    if showstats:
        pt = spss.BasePivotTable(_("Standardize Data"), "Preprocess Standardize")
        pt.TitleFootnotes(_(f"""Mean removal: {TFdict[standardmean]}. Variance scaled: {TFdict[standardstd]}"""))
        cells = [(scaler.mean_[i], scaler.scale_[i]) for i in range(scaler.mean_.size)]

        pt.SimplePivotTable(rowlabels=[spss.CellText.VarName(vardict[v].VariableIndex) for v in varnames], 
            collabels=[_("Mean"), _("Std. Deviation")],
            cells=cells)
    translist.append(scaler)
    return dta

def dorangef(dta, varnames, rangemin, rangemax, showstats, vardict, translist):
    """Standardize data and return data matrix
    rangemin and rangemax define the interval to scale into"""
    
    if None in [rangemin, rangemax]:
        raise ValueError(_("Both the minimum and maximum must be specified"))
    
    scaler = preprocessing.MinMaxScaler(feature_range=(rangemin, rangemax)).fit(dta)
    dta = scaler.fit_transform(dta)
    if showstats:
        pt = spss.BasePivotTable(_("Standardize Data by Min Max"), "Preprocess StandardizeMinMax")
        pt.TitleFootnotes(_(f"""Min: {rangemin}. Max: {rangemax}"""))
        cells = [(scaler.data_min_[i], scaler.data_max_[i], scaler.min_[i], scaler.scale_[i]) for i in range(scaler.n_features_in_)]

        pt.SimplePivotTable(rowlabels=[spss.CellText.VarName(vardict[v].VariableIndex) for v in varnames], 
            collabels=[_("Input Data Minimum"), _("Input Data Maximum"), _("Minimum"), _("Scale")],
            cells=cells)
    translist.append(scaler)
    return dta

def robustf(dta, varnames, center, robustscale, robustqrange, showstats, vardict, translist):
    """Do robust scaling and return data
    
    center specifies centering
    robustscale scales the data to the interquartile range
    robustqrange specifies min and max for scale calculation"""
    
    if len(robustqrange) != 2:
        raise ValueError(_("""If quantile range is specified, two values are required."""))
    
    robustscaler = preprocessing.RobustScaler(with_centering=center,
        with_scaling=robustscale, quantile_range=robustqrange).fit(dta)
    dta = robustscaler.fit_transform(dta)
    if showstats:
        pt = spss.BasePivotTable(_("Robust Data Scaling"), "Preprocess RobustScale")
        rangestr = [str(item) for item in robustqrange]
        pt.TitleFootnotes(
        _(f"""Centering: {TFdict[center]}. Scaling: {TFdict[robustscale]}. Scale range {", ".join(rangestr)}%"""))
        cells = [(robustscaler.scale_[i]) for i in range(robustscaler.scale_.size)]

        pt.SimplePivotTable(rowlabels=[spss.CellText.VarName(vardict[v].VariableIndex) for v in varnames], 
            collabels=["_(Scale)"],
            cells=cells)
    translist.append(robustscaler)
    return dta

def dononlinearf(dta, varnames, nonlinearmethod, nonlinearstandardize, showstats, vardict, translist):
    """Do yeo-johnson or box-cox transformation and return data"""
    
    methods = {"yeojohnson": "yeo-johnson", "boxcox": "box-cox"}
    
    powert = preprocessing.PowerTransformer(method=methods[nonlinearmethod], 
        standardize=nonlinearstandardize).fit(dta)
    dta = powert.fit_transform(dta)
    if showstats:
        pt = spss.BasePivotTable(_("Power Transform Data"), "Preprocess Power")
        pt.TitleFootnotes(_(f"""Method: {methods[nonlinearmethod]}. Standardize: {TFdict[nonlinearstandardize]}"""))
        cells = [(powert.lambdas_[i]) for i in range(powert.lambdas_.size)]
        pt.SimplePivotTable(rowlabels=[spss.CellText.VarName(vardict[v].VariableIndex) for v in varnames], 
            collabels=[_("Lambda")],
            cells=cells)
    translist.append(powert)
    return dta

def quantilef(dta, varnames, quantilenquantiles, quantiledistribution, showstats, vardict, translist):
    """Transform according to quantile parameters
    quantilenquantiles is the number of quantiles
    quantiledistribution is the distribution to aim for"""
    
    quan = preprocessing.QuantileTransformer(n_quantiles=quantilenquantiles,
        output_distribution=quantiledistribution).fit(dta)
    dta = quan.fit_transform(dta)
    if showstats:
        pt = spss.BasePivotTable(_("Quantile Transform (First Five)"), "Preprocess Quantile")
        pt.TitleFootnotes(
        _(f"""N of Quantiles: {quantilenquantiles}. Output Distribution: {quantiledistribution}"""))
        cells = []
        
        for i in  range(0, quan.quantiles_.shape[1]):  # row per quantile
            row = list(quan.quantiles_[:5, i])
            row=row+(5-len(row)) * ["."]   # padding
            cells.extend(row)

        pt.SimplePivotTable(rowlabels=[spss.CellText.VarName(vardict[v].VariableIndex) for v in varnames], 
            collabels=[_("Quantile 1"), _("Quantile 2"), _("Quantile 3"), _("Quantile 4"), _("Quantile 5")],
            cells=cells)
    translist.append(quan)
    return dta

def kbinsf(dta, varnames, kbinsnbins, kbinsbindef, showstats, vardict, translist):
    """Discretize the data and return new dta
    
    kbinsnbins is the number of bins to create
    kbinsbindef is the bin calculation strategy"""
    
    maxcols = min(10, kbinsnbins+1)
    kbins = preprocessing.KBinsDiscretizer(n_bins=kbinsnbins, encode="ordinal", strategy=kbinsbindef).fit(dta)
    dta = kbins.fit_transform(dta)
    if showstats:
        pt = spss.BasePivotTable(_("Bin Data (First Ten Bins)"), "Preprocess Bins")
        pt.TitleFootnotes(_(f"""Number of bins: {kbinsnbins}. Bin strategy: {kbinsbindef}"""))
        cells = []
        for i in range(len(varnames)):
            row = (kbins.bin_edges_[i][:maxcols])
            cells.extend(row)

        pt.SimplePivotTable(rowlabels=[spss.CellText.VarName(vardict[v].VariableIndex) for v in varnames], 
            collabels=[f"Bin Ends {i})" for i in range(maxcols)],
            cells=cells)
    translist.append(kbins)
    return dta
    
    
def createnewdataset(idvar, varnames, dtaid, dta, dataset, datasetnamed, activedataset,
    update, suffix, vardict):
    """Create a new, possibly temporary dataset with variables dtaid, and varnames
    
    idvar is the id variable name
    varnames is the names of the input variables
    dtaid and dta are the values of the idvar and transformed variables
    dataset is the name for the output dataset
    datasetnamed is True if the user supplied a name
    activedataset is the name of the active (input) dataset
    update is true if input variables should be overwritten
    in the input dataset
    suffix is the name to be appended if not update
    vardict is a variable dictionary"""
    

    spss.EndProcedure()
    
    
    # create new dataset for output
    curs = Spssdata(accessType="n", maxaddbuffer=len(varnames) * 8 + vardict[idvar].VariableType)
    curs.append(vdef(idvar, vtype=vardict[idvar].VariableType))
    for vname in varnames:
        curs.append(vdef(vname))
    curs.commitdict()

    # create cases for transformed variables
    # Numpy nan's must be converted to None for SPSS
    for i in range(dta.shape[0]):
        curs.appendvalue(idvar, dtaid[i])
        for j, v in enumerate(varnames):
            val = dta[i, j]
            if np.isnan(val):
                val = None
            curs.appendvalue(v, val)
        curs.CommitCase()
    curs.CClose()
    spss.Submit(f"""DATASET NAME {dataset}.""")
    
    # get variable properties from input dataset but not missing
    # value definitions and value labels
    varstr = " ".join([idvar]+varnames)
    varnamesstr = " ".join(varnames)
    cmd = f"""APPLY DICTIONARY FROM {activedataset}
    /TARGET VARIABLES= {varstr}
    /VARINFO ALIGNMENT ATTRIBUTES FORMAT LEVEL ROLE VARLABEL WIDTH."""
    spss.Submit(cmd)
    
    # rename output variables if suffix
    if suffix:
        newnames = " ".join([vname + "_" + suffix for vname in varnames])
        rcmd = f"""DATASET ACTIVATE {dataset}.
        RENAME VARIABLES ({varnamesstr} = {newnames})"""
        spss.Submit(rcmd)    
    
    # merge files if requested.  Existing variables will have
    # values updated, and any new variables will be created.
    # UPDATE will fail if case id's are not monotonic.
    if update:
        cmd =  f"""DATASET ACTIVATE {activedataset}.
    UPDATE /FILE=*
    /FILE={dataset}
    /BY {idvar}"""
        spss.Submit(cmd)
        # remove missing value and value labels definitions as they are obsolete
        spss.Submit(f"""MISSING VALUES {varnamesstr} ().
        VALUE LABELS {varnamesstr}.""")
    
    if not datasetnamed:
        spss.Submit(f"""DATASET CLOSE {dataset}.
    DATASET ACTIVATE {activedataset}.""")


def applytrans(dta, intransfile, varnames, showstats, translist):
    """Apply saved transformation specifications and return data
    
    dta is the data to transform
    intransfile is the pickled set of transformations, already read
    showstats indicates whether to display transformation parameter data
    translist is the contents of intransfile"""
    
    print(_(f"Applying transformations from file {intransfile} of {translist[0]}"))
        
    if showstats:
        print(_("Variables:"), " ".join(translist[2]))
    
    for trans in translist[3:]:
        if showstats:
            params = trans.get_params()
            print(_("Transformation:"), trans, params)
        dta = trans.fit_transform(dta)
    return dta


def  Run(args):
    """Execute the STATS PREPROCESS command"""

    args = args[list(args.keys())[0]]

    oobj = Syntax([
        Template("VARIABLES", subc="",  ktype="existingvarlist", var="varnames", islist=True),
        Template("DATASET", subc="", ktype="existingvarlist", var="dataset", islist=False), 
        Template("UPDATE", subc="", ktype="bool", var="update"),
        Template("ID", subc="", ktype="existingvarlist", var="idvar", islist=False),
        Template("SUFFIX", subc="", ktype="varname", var="suffix"),
        Template("SORT", subc="", ktype="bool", var="sorting"),
        Template("SAVETRANS", subc="", ktype="literal", var="outtransfile"),
        Template("APPLYTRANS", subc="", ktype="literal", var="intransfile"), 
        
        Template("PRINT", subc="OPTIONS", ktype="bool", var="showstats"),
        
        Template("DOSTD", subc="STANDARD", ktype="bool", var="dostd"),      # StandardScaler
        Template("MEAN", subc="STANDARD", ktype="bool", var="standardmean"),
        Template("STD", subc="STANDARD", ktype="bool", var="standardstd"),
        
        Template("DORANGE", subc="RANGE", ktype="bool", var="dorange"),     # MinMaxScaler
        Template("MIN", subc="RANGE", ktype="float", var="rangemin"), 
        Template("MAX", subc="RANGE", ktype="float", var="rangemax"),
        Template("MAXABS", subc="RANGE", ktype="bool", var="rangemaxabs"),       # MaxAbsScaler (not with min, max)
        
        Template("DOROBUST", subc="ROBUST", ktype="bool", var="dorobust"),  # RobustScaler
        Template("CENTER", subc="ROBUST", ktype="bool", var="robustcenter"),
        Template("SCALE", subc="ROBUST", ktype="bool", var="robustscale"),  # scale to IQR
        Template("QRANGE", subc="ROBUST", ktype="float", var="robustqrange", islist=True, 
            vallist=[0., 100.]),                                             # two values or omit
        
        Template("DONONLINEAR", subc="NONLINEAR", ktype="bool", var="dononlinear"),     # PowerTransformer
        Template("METHOD", subc="NONLINEAR", ktype="str", var="nonlinearmethod",
            vallist=["boxcox", "yeojohnson"]),
        Template("STANDARDIZE", subc="NONLINEAR", ktype="bool", var="nonlinearstandardize"), 
        
        Template("DOQUANTILE", subc="QUANTILE", ktype="bool", var="doquantile"),        # QuantileTransformer
        Template("NQUANTILES", subc="QUANTILE", ktype="int", var="quantilenquantiles",
            vallist=[1, ]),
        Template("DISTRIBUTION", subc="QUANTILE", ktype="str", var="quantiledist",
            vallist=["uniform", "normal"]), 

        
        Template("DOKBINS", subc="KBINS", ktype="bool", var="dokbins"),
        Template("NBINS", subc="KBINS", ktype="int", var="kbinsnbins",
            vallist=[2, ]),
        Template("BINDEF", subc="KBINS", ktype="str", var="kbinsbindef",
            vallist=["uniform", "quantile", "kmeans"])])
        
    ### Template("DOIMPUTATION", subc="IMPUTATION", ktype="bool", var="doimputation"),  # TODO
        
        
    #enable localization
    global _
    try:
        _("---")
    except:
        def _(msg):
            return msg

    # A HELP subcommand overrides all else
    if "HELP" in args:
        #print helptext
        helper()
    else:
        processcmd(oobj, args, doprep)

def helper():
    """open html help in default browser window
    
    The location is computed from the current module name"""
    
    import webbrowser, os.path
    
    path = os.path.splitext(__file__)[0]
    helpspec = "file://" + path + os.path.sep + \
         "markdown.html"
    
    # webbrowser.open seems not to work well
    browser = webbrowser.get()
    if not browser.open_new(helpspec):
        print(("Help file not found:" + helpspec))
try:    #override
    from extension import helper
except:
    pass        
