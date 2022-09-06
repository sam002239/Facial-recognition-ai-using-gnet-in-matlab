factrain=imageDatastore("train","IncludeSubfolders",true,"LabelSource","foldernames");
auds=augmentedImageDatastore([224 224],factrain,"ColorPreprocessing","gray2rgb");
factest=imageDatastore("validation","IncludeSubfolders",true,"LabelSource","foldernames");
numclasses=numel(categories(factrain.Labels));
audst=augmentedImageDatastore([224 224],factest,"ColorPreprocessing","gray2rgb")
 
options=trainingOptions("sgdm","InitialLearnRate",0.001,"ExecutionEnvironment","multi-gpu")
[enabled, mode] = Simulink.sdi.isPCTSupportEnabled
%%

[expresion,info]=trainNetwork(auds,lgraph_1,options)
testpreds=classify(expresion,audst)






