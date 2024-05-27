%% CODICE 

clear all, close all, clc

%% Loading the data
% Inserting the datapath for the directory with the subjects' data named
% 'P1', 'P2', ..., 'Pn' with n=#subjects
%datapath=input('Insert datapath: ');
datapath='C:\Users\Francesca\Documents\MATLAB\NeuroRobotics\Project\Data';
%datapath='/Users/statosh/Desktop/UNIPD/1st semester/Neurorobotics and neurorehabilitation/project/Data';
%datapath = '/Users/ceciliarossi/Desktop/MAGISTRALE/Neurobotics and Neurehabilitation/Progetto/Codice/Data';

%% Create a Structure in which saving the results of classification
nsubjects = 5;

% Threshold for Fisher score
threshold = 0; %0.5;
if threshold == 0
    subj_score_RF = cell(1,nsubjects);
    subj_score_LDA = cell(1,nsubjects);
    subj_score_QDA = cell(1,nsubjects);
    subj_score_SVM = cell(1,nsubjects);
else
    subj_score_fisher_LDA = cell(1,nsubjects);
    subj_score_fisher_RF = cell(1,nsubjects);
    subj_score_fisher_QDA = cell(1,nsubjects);
    subj_score_fisher_SVM = cell(1,nsubjects);
end

%%
spatial_filter="CCA";
% "CAR" for CAR
% "CCA" for CCA

GA_err_matrix={};
GA_corr_matrix={};
GA_err_vis_matrix={};
GA_corr_vis_matrix={};
GA_err1_vis_matrix={};
GA_err2_vis_matrix={};
labels={};
eventi_tot={};
features_tot={};
TrainTestSignals={};

for sId=1:nsubjects
    % Selecting the subject to analyse
    disp(['Analysis for subject: ' num2str(sId)]);
    selSubject=sId;
    
    % Selecting the corresponding directory with the selected subject's data
    newDataPath=fullfile(datapath, ['P' num2str(selSubject)]);
    elementi=dir(newDataPath);
    nfiles=length(elementi);
    
    % Removing the folder references
    indici_dir=[];
    
    for i=1:nfiles
        cfilename=elementi(i).name;
        if elementi(i).isdir==1
            indici_dir=[indici_dir, i];
        end
    end
    elementi(indici_dir)=[];
    
    %Loading and concatenating files
    Modality="All"; %"Offline" if one wants to analyse only offline files --> all of them are offline thought
    [s, TYP, DUR, POS, SampleRate, Mk, Rk]=conc_project(elementi, newDataPath, Modality);
    DUR=DUR';
    
    % Creating the struct for the labelling function
    eventi.POS=POS;
    eventi.DUR=DUR;
    eventi.TYP=TYP;
    
    nsamples=size(s,1); % s is a matrix [samples x channels];
    
    % Labelling
    [RaceStart, RaceEnd, TurnLeftSession, TurnRightSession, LightSession, StraightSession, CommLeft, CommRight, comm_start, idx_comm]=labelling_project(nsamples, eventi);
    
    % Removing the informations related to the 17th channel --> GND channel
    s=s(:,[1:16]);
    
    nruns=length(unique(Rk));
    
    % Strategy LEAVE-ONE-OUT for Classification
    score_tabs_RF = cell(1,nruns);
    score_tabs_LDA = cell(1,nruns);
    score_tabs_QDA = cell(1,nruns);
    score_tabs_SVM = cell(1,nruns);
        
    %% PROCESSING: APPLYING SPATIAL FILTER
    % Removing the informations related to the 17th channel --> GND channel
    s=s(:,1:16);
    nchannels=size(s,2);
    
    if spatial_filter=="CAR"
        s_filt=s-repmat(mean(s,2), [1 nchannels]); %CAR
    elseif spatial_filter=="CCA"
        s_filt=s;
        % the CCA spatial filter will be applied later in the code
    else
        error('Unknown spatial filtering method')
    end
    
    
    %% BUTTERWORTH FILTER
    % Filtering the signal in [1 10] Hz band
    band=[1 10];
    order=4; %Order of the Butterworth filter
    Fc=SampleRate; %Hz
    Wn=band/(Fc/2);
    
    [b,a]=butter(order,Wn);
    
    s_filtered=nan(nsamples,nchannels);
    for chId = 1:nchannels
        s_filtered(:, chId) = filtfilt(b, a, s_filt(:, chId));
    end
    
    media_s=mean(s_filtered);
    media_matrice=repmat(media_s, [size(s_filtered,1), 1]);
    
    
    %% SUBSAMPLING FROM 512 HZ TO 64 HZ
    nsurviving_samples=floor(nsamples/8);
    
    s_downsampled=nan(nsurviving_samples,nchannels);
    Rk_downsampled=nan(1,nsurviving_samples);
    
    k=1;
    for cId=1:nchannels
        for p=1:nsurviving_samples
            s_downsampled(p,cId)=s_filtered(k,cId);
            k=k+8;
        end
        k=1;
    end
    
    % Repeating the same downsampling operation for Rk
    k=1;
    for rId=1:nsurviving_samples
        Rk_downsampled(rId)=Rk(k);
        RaceStart(rId)=RaceStart(k);
        RaceEnd(rId)=RaceEnd(k);
        TurnLeftSession(rId)=TurnLeftSession(k);
        TurnRightSession(rId)=TurnRightSession(k);
        LightSession(rId)=LightSession(k);
        StraightSession(rId)=StraightSession(k);
        CommLeft(rId)=CommLeft(k);
        CommRight(rId)=CommRight(k);
        k=k+8;
    end
    
    % Converting the POSitions and DURations from the 512Hz sampling to the
    % 64Hz sampling
    newPOS=nan(length(POS),1);
    newDUR=nan(length(POS),1);
    
    index_pos=1:8:nsamples; %indexes of the position of every surviving sample
    for pId=1:length(POS)
        if ismember(POS(pId),index_pos)
            newPOS(pId)=(POS(pId)-1)/8;
        else
            ind=find(index_pos>POS(pId),1);
            newPOS(pId)=(index_pos(ind)-1)/8;
        end
        % We are not converting TYP because the only indexes with duration
        % inferior to 8 samples are TYP=800 or TYP=8800, that are not further
        % considered in our analysis
    end
    
    for dId=2:length(POS)
        newDUR(dId-1)=newPOS(dId)-newPOS(dId-1);
    end
    newDUR(dId)=nsurviving_samples-newPOS(dId);
    
    eventi.POS=newPOS; eventi.DUR=newDUR;
    
    % Converting also the positions in comm_start (starting samples of every
    % command sent)
    for pId=1:length(comm_start)
        if ismember(comm_start(pId),index_pos)
            comm_start(pId)=(comm_start(pId)-1)/8;
        else
            ind=find(index_pos>comm_start(pId),1);
            comm_start(pId)=(index_pos(ind)-1)/8;
        end
    end
    
    
    %% EXTRACTING CHANNELS
    % I'm selecting channels Cz and FCz
    % chanLabels={'FCz', 'Cz'};
    % chanSel=[4,9];
    
    chanLabels={'Fz','FCz', 'Cz','CPz'};
    chanSel=[1,4,9,14];
    
    NewSampleRate=64; %Hz
    
    time_window=[0.2 1]; %s
    % Identifying the samples marking the beginning and the end of the
    % time window
    s_in=floor(NewSampleRate*time_window(1));
    s_end=floor(NewSampleRate*time_window(2));
    extr_samples=s_end-s_in+1;
    ntrials=length(comm_start);
    trial_runs=nan(1,ntrials);
    nselChans=length(chanSel);
    
    %%  Creating the feature matrix
    % Creating the feature matrix = [extracted samples x selected channels x trials]
    % The trial starts when the command is sent
    FeatureData=nan(extr_samples, nselChans, ntrials);
    
    for cId=1:nselChans
        for nId=1:ntrials
            trial_start=comm_start(nId);
            curr_run=unique(Rk_downsampled(trial_start+s_in:trial_start+s_end));
            trial_runs(nId)=curr_run;
            FeatureData(:, cId, nId)=s_downsampled(trial_start+s_in:trial_start+s_end, chanSel(cId));
        end
    end
    
    
    ntrials=size(FeatureData,3);
    nruns=length(unique(Rk_downsampled));
    
    % To extract the features corresponding to each run, use the function
    % 'extract_runs';
    
    %% Creating an Array with all the run matrices
    RunArray=cell(1,nruns);
    for rId=1:nruns
        RunArray{rId}=extract_runs(FeatureData, trial_runs, rId);
    end
    
    if spatial_filter=="CAR"
        %% GRAND AVERAGE IN THE TIME DOMAIN
        % Setting the struct to use inside extractLabels2;
        mGA.FeaturesData=FeatureData;
        mGA.comm_start=comm_start;
        etichette = extractLabels2(mGA, eventi);
        labels{sId}=etichette;
        
        ErrData=FeatureData(:,:,etichette~=0);
        CorrectData=FeatureData(:,:, etichette==0);
        
        GA_err=mean(ErrData, 3);
        GA_corr=mean(CorrectData,3);
        GA_err_matrix{sId}=GA_err;
        GA_corr_matrix{sId}=GA_corr;
        
        %Creating time references
        newTc=1/NewSampleRate;
        t_err=[0: newTc: (extr_samples-1)*newTc];
        
        %% Visualizing the Grand Average for the selected subject
        
        % Visualizing the signal from the command start to 1 sec
        time_visualization=[0 1]; %s
        VisualizationData=nan(s_end, nselChans, ntrials);
        for cId=1:nselChans
            for nId=1:ntrials
                trial_start=comm_start(nId);
                VisualizationData(:, cId, nId)=s_downsampled(trial_start:trial_start+s_end-1, chanSel(cId));
            end
        end
        GA_err_vis=mean(VisualizationData(:,:,etichette~=0), 3);
        GA_corr_vis=mean(VisualizationData(:,:,etichette==0), 3);
        GA_err_vis_matrix{sId}=GA_err_vis;
        GA_corr_vis_matrix{sId}=GA_corr_vis;
        
        % Time vector
        t_err_vis=[0: newTc: (s_end-1)*newTc];
        
        %Setting the y axis
        max_values=nan(1,nselChans);
        for cId=1:nselChans
            ymax=max(abs(GA_err_vis(:,cId)));
            max_values(cId)=ymax;
        end
        max_amp=max(max_values);
        
        figure()
        for cId=1:nselChans
            subplot(nselChans/2,2,cId)
            plot(t_err_vis,GA_err_vis(:,cId), 'r-', 'Linewidth', 1)
            hold on
            plot(t_err_vis, GA_corr_vis(:,cId), 'b-', 'Linewidth', 1)
            xlim([0 t_err_vis(end)])
            ylim([(-max_amp-0.1) (max_amp+0.1)])
            grid on
            xline(0.2, 'm--', 'Linewidth', 1);
            xlabel('time [s]')
            ylabel('Amplitude [\muV]')
            title(['Grand Average for channel ' chanLabels{cId} ' - subject ' num2str(selSubject)])
        end
        sgtitle('Grand Average')
        legend('GA error', 'GA correct', 'trial')
        
        %% Visualizing the Grand Average for the selected subject differentiating between the types of error
        GA_err1_vis=mean(VisualizationData(:,:,etichette==1), 3);
        GA_err2_vis=mean(VisualizationData(:,:,etichette==-1), 3);
        GA_err1_vis_matrix{sId}=GA_err1_vis;
        GA_err2_vis_matrix{sId}=GA_err2_vis;
        
        %Setting the y axis
        max_values=[];
        for cId=1:nselChans
            y1max=max(abs(GA_err1_vis(:,cId)));
            y2max=max(abs(GA_err2_vis(:,cId)));
            max_values=[max_values, y1max, y2max];
        end
        max_amp=max(max_values);
        
        figure()
        for cId=1:nselChans
            subplot(nselChans/2,2,cId)
            plot(t_err_vis,GA_err1_vis(:,cId), 'r-', 'Linewidth', 1)
            hold on
            plot(t_err_vis,GA_err2_vis(:,cId), 'm-', 'Linewidth', 1)
            plot(t_err_vis, GA_corr_vis(:,cId), 'c-', 'Linewidth', 1)
            xlim([0 t_err_vis(end)])
            ylim([(-max_amp-0.1) (max_amp+0.1)])
            grid on
            xline(0.2, 'b--', 'Linewidth', 1);
            xlabel('time [s]')
            ylabel('Amplitude [\muV]')
            title(['Grand Average for channel ' chanLabels{cId} ' - subject ' num2str(selSubject)])
        end
        sgtitle('Grand Average distinguishing the types of ErrP')
        legend('GA curve error', 'GA straight/light error', 'GA correct', 'trial onset')
    end
    
    for runId=1:nruns
        FeatureTest=RunArray{runId};
        run_test=runId;
        ntrials_mancanti=size(FeatureTest, 3);
        tempRun=RunArray;
        tempRun{runId}={};
        FeatureTrain=[];
        
        for ruId=1:nruns
            if sum(size(tempRun{ruId})) ~=0
                FeatureTrain=cat(3, FeatureTrain,tempRun{ruId});
            end
        end
        
        %% CANONICAL CORRELATION ANALYSIS
        ntrials=length(comm_start);
        
        if spatial_filter=="CCA"
            %Creating the matrix with segmented epochs
            epoch_window=[0.2 1]; % s, with respect to the onset of trials
            % It's convenient for us to take the time range from [0 1] s
            % instead of [0.2 1] s.
            sample_in=floor(NewSampleRate*epoch_window(1));
            sample_end=floor(NewSampleRate*epoch_window(2));
            nsamples_epoch=abs(sample_end)-abs(sample_in)+1;
            [Wx, U, TrainSet_filtered]=CCA_filtering(FeatureData, FeatureTrain, eventi, comm_start, trial_runs, run_test);
            
            % Changing the structure of FeatureTest
            test_data=nan(size(FeatureTest,2), nsamples_epoch*size(FeatureTest, 3));
            for cId=1:size(FeatureTest,2)
                k=0;
                for trId=1:size(FeatureTest, 3)
                    test_data(cId,k*nsamples_epoch+1:trId*nsamples_epoch)=FeatureTest(:,cId,trId);
                    k=k+1;
                end
            end
            TestSet_filt=Wx'*test_data;
            
            TestSet_filtered=nan(nsamples_epoch, nselChans, size(FeatureTest,3));
            for cId=1:nselChans
                k=0;
                for nId=1:size(FeatureTest,3)
                    TestSet_filtered(:, cId, nId)=TestSet_filt(cId, k*nsamples_epoch+1:nId*nsamples_epoch);
                    k=k+1;
                end
            end
            
            %% VISUALIZATION OF THE CCA FILTERING FOR ONE OF THE RUNS
            if runId==8
                mGA.FeaturesData=FeatureData;
                mGA.comm_start=comm_start;
                etichette = extractLabels2(mGA, eventi);
                % etichette=etichette(trial_runs~=run_test);
                etichette=etichette(trial_runs==run_test);
                
                %ErrData=FeatureTrain(:,:,etichette~=0);
                ErrData=FeatureTest(:,:,etichette~=0);
                %CorrectData=FeatureTrain(:,:, etichette==0);
                CorrectData=FeatureTest(:,:, etichette==0);
                
                GA_err=mean(ErrData, 3);
                GA_corr=mean(CorrectData,3);
                GA_err_matrix{sId}=GA_err;
                GA_corr_matrix{sId}=GA_corr;
                
                %Creating time references
                newTc=1/NewSampleRate;
                t_err=[0: newTc: (extr_samples-1)*newTc];
                
                %% Visualizing the Grand Average for the selected subject
                
                % Visualizing the signal from the command start to 1 sec
                time_visualization=[0 1]; %s
                GA_err_vis=GA_err;
                GA_corr_vis=GA_corr;
                GA_err_vis_matrix{sId}=GA_err_vis;
                GA_corr_vis_matrix{sId}=GA_corr_vis;
                
                % Time vector
                t_err_vis=[0.2: newTc: (size(FeatureTrain,1)-1)*newTc+0.2];
                
                %Setting the y axis
                max_values=nan(1,nselChans);
                for cId=1:nselChans
                    ymax=max(abs(GA_err_vis(:,cId)));
                    max_values(cId)=ymax;
                end
                max_amp=max(max_values);
                
                figure()
                for cId=1:nselChans
                    subplot(nselChans/2,2,cId)
                    plot(t_err_vis,GA_err_vis(:,cId), 'r-', 'Linewidth', 1)
                    hold on
                    plot(t_err_vis, GA_corr_vis(:,cId), 'b-', 'Linewidth', 1)
                    ylim([(-max_amp-0.1) (max_amp+0.1)])
                    grid on
                    xlim([0.2, 1])
                    xticks([0.2:0.1:1])
                    xlabel('time [s]')
                    ylabel('Amplitude [\muV]')
                    title(['Grand Average for channel ' chanLabels{cId} ' - subject ' num2str(selSubject)])
                end
                sgtitle(['Grand Average for run ' num2str(runId)])
                legend('GA error', 'GA correct')
                
                %% Visualizing the Grand Average for the selected subject differentiating between the types of error
                %GA_err1_vis=mean(FeatureTrain(:,:,etichette==1), 3);
                GA_err1_vis=mean(FeatureTest(:,:,etichette==1), 3);
                %GA_err2_vis=mean(FeatureTrain(:,:,etichette==-1), 3);
                GA_err2_vis=mean(FeatureTest(:,:,etichette==-1), 3);
                GA_err1_vis_matrix{sId}=GA_err1_vis;
                GA_err2_vis_matrix{sId}=GA_err2_vis;
                
                %Setting the y axis
                max_values=[];
                for cId=1:nselChans
                    y1max=max(abs(GA_err1_vis(:,cId)));
                    y2max=max(abs(GA_err2_vis(:,cId)));
                    max_values=[max_values, y1max, y2max];
                end
                max_amp=max(max_values);
                
                figure()
                for cId=1:nselChans
                    subplot(nselChans/2,2,cId)
                    plot(t_err_vis,GA_err1_vis(:,cId), 'r-', 'Linewidth', 1)
                    hold on
                    plot(t_err_vis,GA_err2_vis(:,cId), 'm-', 'Linewidth', 1)
                    plot(t_err_vis, GA_corr_vis(:,cId), 'c-', 'Linewidth', 1)
                    ylim([(-max_amp-0.1) (max_amp+0.1)])
                    grid on
                    xlim([0.2, 1])
                    xticks([0.2:0.1:1])
                    xlabel('time [s]')
                    ylabel('Amplitude [\muV]')
                    title(['Grand Average for channel ' chanLabels{cId} ' - subject ' num2str(selSubject)])
                end
                sgtitle(['Grand Average distinguishing the types of ErrP for run ' num2str(runId)])
                legend('GA curve error', 'GA straight/light error', 'GA correct')
            end
            
            mGA.FeaturesData=FeatureData;
            mGA.comm_start=comm_start;
            etichette = extractLabels2(mGA, eventi);
        elseif spatial_filter=="CAR"
            TrainSet_filtered = FeatureTrain;
            TestSet_filtered = FeatureTest;
        else
            disp('Error in the definition of the spatial filtering method');
        end
        
        TrainTestSignals{sId, runId, 1}=TrainSet_filtered;
        TrainTestSignals{sId, runId, 2}=TestSet_filtered;
        
        %% Saving variables in a struct
        features.FeaturesData=FeatureData; % matrix with the selected features
        features.ChansSel=chanSel; % channels selected
        features.ChanLabels=chanLabels; %labels of selected channels
        features.Array=RunArray;
        features.comm_start=comm_start; % starting positions of every command
        features.idx_comm=idx_comm; % index of events where useful commands where sent
        features.labels=labels;
        features.filtro=spatial_filter;
        features.TestTrainSignals=TrainTestSignals;
        eventi.s=s_downsampled; % processed signal
        eventi.SampleRate=NewSampleRate; % new sample rate
        eventi.RaceStart=RaceStart;
        eventi.RaceEnd=RaceEnd;
        eventi.TurnLeftSession=TurnLeftSession;
        eventi.TurnRightSession=TurnRightSession;
        eventi.LightSession=LightSession;
        eventi.StraightSession=StraightSession;
        eventi.CommLeft=CommLeft;
        eventi.CommRight=CommRight;
        
        eventi_tot{sId}=eventi;
        features_tot{sId}=features;
        
        %% Saving the corresponding results
        datapath_results=fullfile(datapath,'Results/');
        name_file=fullfile(datapath_results, ['P' num2str(selSubject) '_features.mat']);
        save(name_file, 'features', 'eventi');


        
        
        %% CLASSIFICATION %%%%%%%%%%%%%%%%%%%
        %% STRATEGY LEAVE-ONE-OUT
        
        nchannels = 4;
        nfeatures = 24;
        
        %% Features and Labels Extraction for Train and Test Set
        
        % We only considered features related to the signal amplitude:
        %  1. Positive Peak (max value in the window)
        %  2. Negative Peak (min value in the window)
        %  3. Latency of the Positive Peak
        %  4. Latency of the Negative Peak
        %  5. Mean of the signal in the window
        %  6. Standard Deviation of the Signal in the window 
        % The features from 1 to 6 are referred to Channel 1 = FCz Channel
        % The features from 7 to 12 are referred to Channel 2 = Cz Channel
        % The features from 13 to 18 are referred to Channel 3 = ...
        % The features from 19 to 24 are referred to Channel 4

        % y: is a a vector (1 x ntrials in that run) containing the labels for every trial in the run 
        % ( correct = 0; error = 1(curve error) or -1(straight/light error) ) 
       
        X_train=nan(nfeatures, size(TrainSet_filtered, 3));
        X_test=nan(nfeatures, size(TestSet_filtered, 3));
        ntrials_train = size(TrainSet_filtered, 3);
        ntrials_test = size(TestSet_filtered,3);
        
        %Extracting the labels
        num_t = 0;
        y = cell(1,nruns);
        for run = 1:nruns
            ntrials = size(features.Array{run}, 3);
            lab_run = nan(1,ntrials);
            for trial=1:ntrials
                num_t = num_t + 1;
                lab_run(trial) = etichette(num_t);
            end %for trials
            y{run} = lab_run;
        end %for 
        
        y_test = y{runId};
        y_train = [];
        
        for n_r=1:nruns
            if n_r ~= runId
                y_train = [y_train y{n_r}];
            end %if
        end %for
        y_train = abs(y_train)'; y_test = abs(y_test)';
        
        % Extracting the features for the train
        for trial=1:ntrials_train
            for ch=1:nchannels
                % positive peak calculation
                trial_sig_train = TrainSet_filtered(:, ch, trial);
                [pos_peak, idx_pos] = max(trial_sig_train);
                [neg_peak, idx_neg] = min(trial_sig_train);
                mean_sig=mean(trial_sig_train);
                std_sig=std(trial_sig_train);
                % neg. and pos. peak latency
                pos_latency=0.2 + idx_pos/SampleRate;
                neg_latency=0.2 + idx_neg/SampleRate;
                % Just smart slicing
                X_train((nfeatures/nchannels * (ch-1))+1:ch*nfeatures/nchannels, trial) = [pos_peak, neg_peak,  pos_latency, neg_latency, mean_sig, std_sig];
            end %for channels
        end %for trials
        X_train = X_train';
        
        % Extracting the features for the test
        for trial=1:ntrials_test
            for ch=1:nchannels
                % positive peak calculation
                trial_sig_test = TestSet_filtered(:, ch, trial);
                [pos_peak, idx_pos] = max(trial_sig_test);
                [neg_peak, idx_neg] = min(trial_sig_test);
                mean_sig=mean(trial_sig_test);
                std_sig=std(trial_sig_test);
                % neg. and pos. peak latency
                pos_latency=0.2 + idx_pos/SampleRate;
                neg_latency=0.2 + idx_neg/SampleRate;
                % Just smart slicing
                X_test((nfeatures/nchannels * (ch-1))+1:ch*nfeatures/nchannels, trial) = [pos_peak, neg_peak,  pos_latency, neg_latency, mean_sig, std_sig];
            end %for channels
        end %for trials
        X_test = X_test';
        
        %% %% CALIBRATION PHASE
        %% Training a Random Forest Classifier
        nTrees=1;
        B = TreeBagger(nTrees,X_train,y_train, 'Method', 'classification'); 

        %% Training of a LDA Classificator
        LDAmodel = fitcdiscr(X_train, y_train, 'ClassNames', [0,1], 'DiscrimType', 'linear');
        
        %% Training a QDA Classificator
        QDAmodel = fitcdiscr(X_train, y_train, 'ClassNames', [0,1], 'DiscrimType', 'quadratic');
        
        %% Training a SVM Classificator
        SVMmodel = fitcsvm(X_train, y_train, 'ClassNames', [0,1]);
        
        %% %% EVALUATION PHASE
        %% Compute the Test Error and TN,TP,FP,FN on test
                
        % Predict Labels for RF
        [predicted_labels_test_RF, score_test_RF, cost_test_RF] = predict(B, X_test);
        predicted_labels_test_RF = str2double(predicted_labels_test_RF);
        [TP_test_RF, TN_test_RF, FN_test_RF, FP_test_RF, error_test_RF] = modelResults(y_test, predicted_labels_test_RF);
        
        % Predict Labels for LDA
        [predicted_labels_test_LDA, score_test_LDA, cost_test_LDA] = predict(LDAmodel, X_test);
        [TP_test_LDA, TN_test_LDA, FN_test_LDA, FP_test_LDA, error_test_LDA] = modelResults(y_test, predicted_labels_test_LDA);
        
%         % Predict Labels for QDA
         [predicted_labels_test_QDA, score_test_QDA, cost_test_QDA] = predict(QDAmodel, X_test);
         [TP_test_QDA, TN_test_QDA, FN_test_QDA, FP_test_QDA, error_test_QDA] = modelResults(y_test, predicted_labels_test_QDA);
        
        %Predict Labels for SVM
        [predicted_labels_test_SVM,score_test_SVM,cost_test_SVM] = predict(SVMmodel, X_test);
        [TP_test_SVM, TN_test_SVM, FN_test_SVM, FP_test_SVM, error_test_SVM] = modelResults(y_test, predicted_labels_test_SVM);
        
        
        %% Compute the Train Error and TN, TP, FP, FN on train
        
        % Predict Labels for RF
        [predicted_labels_train_RF, score_train_RF, cost_train_RF] = predict(B, X_train);
        predicted_labels_train_RF = str2double(predicted_labels_train_RF);
        [TP_train_RF, TN_train_RF, FN_train_RF, FP_train_RF, error_train_RF] = modelResults(y_train, predicted_labels_train_RF);
        
        % Predict Labels for LDA
        [predicted_labels_train_LDA, score_train_LDA, cost_train_LDA] = predict(LDAmodel, X_train);
        [TP_train_LDA, TN_train_LDA, FN_train_LDA, FP_train_LDA, error_train_LDA] = modelResults(y_train, predicted_labels_train_LDA);
        
%         % Predict Labels for QDA
        [predicted_labels_train_QDA, score_train_QDA, cost_train_QDA] = predict(QDAmodel, X_train);
        [TP_train_QDA, TN_train_QDA, FN_train_QDA, FP_train_QDA, error_train_QDA] = modelResults(y_train, predicted_labels_train_QDA);
        
        %Predict Labels for SVM
        [predicted_labels_train_SVM,score_train_SVM,cost_train_SVM] = predict(SVMmodel, X_train);
        [TP_train_SVM, TN_train_SVM, FN_train_SVM, FP_train_SVM, error_train_SVM] = modelResults(y_train, predicted_labels_train_SVM);
        
        %% Adding all the evaluation on Score_tabs
        score_tabs_RF = resultStruct(score_tabs_RF, runId, error_train_RF, error_test_RF, TP_train_RF, TN_train_RF, FN_train_RF, FP_train_RF, TP_test_RF, TN_test_RF, FP_test_RF, FN_test_RF, y_train, y_test, score_train_RF, score_test_RF);
        score_tabs_LDA = resultStruct(score_tabs_LDA, runId, error_train_LDA, error_test_LDA, TP_train_LDA, TN_train_LDA, FN_train_LDA, FP_train_LDA, TP_test_LDA, TN_test_LDA, FP_test_LDA, FN_test_LDA, y_train, y_test, score_train_LDA, score_test_LDA);
        score_tabs_QDA = resultStruct(score_tabs_QDA, runId, error_train_QDA, error_test_QDA, TP_train_QDA, TN_train_QDA, FN_train_QDA, FP_train_QDA, TP_test_QDA, TN_test_QDA, FP_test_QDA, FN_test_QDA, y_train, y_test, score_train_QDA, score_test_QDA);
        score_tabs_SVM = resultStruct(score_tabs_SVM, runId, error_train_SVM, error_test_SVM, TP_train_SVM, TN_train_SVM, FN_train_SVM, FP_train_SVM, TP_test_SVM, TN_test_SVM, FP_test_SVM, FN_test_SVM, y_train, y_test, score_train_SVM, score_test_SVM);

        %% Plotting the ROC Curve 
%         figure()
%         subplot(211)
%         plot(score_tabs_RF{1, run}.Xroc_train, score_tabs_RF{1, runId}.Yroc_train); title(['ROC in Train for RF Classificator for subj: ', num2str(subj), 'and run: ', num2str(run)])
%         xlabel('False Positive Rate'); ylabel('True Positive Rate')
%         subplot(212)
%         plot(score_tabs_RF{1, run}.Xroc_test, score_tabs_RF{1, runId}.Yroc_test); title('ROC in Test for RF Classificator for subj: ', num2str(subj), 'and run: ', num2str(run)])
%         xlabel('False Positive Rate'); ylabel('True Positive Rate')
%         
%         figure()
%         subplot(211)
%         plot(score_tabs_LDA{1, run}.Xroc_train, score_tabs_LDA{1, runId}.Yroc_train); title('ROC in Train for LDA Classificator for subj: ', num2str(subj), 'and run: ', num2str(run)])
%         xlabel('False Positive Rate'); ylabel('True Positive Rate')
%         subplot(212)
%         plot(score_tabs_LDA{1, run}.Xroc_test, score_tabs_LDA{1, runId}.Yroc_test); title('ROC in Test for LDA Classificator for subj: ', num2str(subj), 'and run: ', num2str(run)])
%         xlabel('False Positive Rate'); ylabel('True Positive Rate')
%         
%         figure()
%         subplot(211)
%         plot(score_tabs_QDA{1, run}.Xroc_train, score_tabs_QDA{1, runId}.Yroc_train); title('ROC in Train for QDA Classificator for subj: ', num2str(subj), 'and run: ', num2str(run)])
%         xlabel('False Positive Rate'); ylabel('True Positive Rate')
%         subplot(212)
%         plot(score_tabs_QDA{1, run}.Xroc_test, score_tabs_QDA{1, runId}.Yroc_test); title('ROC in Test for QDA Classificator for subj: ', num2str(subj), 'and run: ', num2str(run)])
%         xlabel('False Positive Rate'); ylabel('True Positive Rate')
%         
%         figure()
%         subplot(211)
%         plot(score_tabs_LDA{1, run}.Xroc_train, score_tabs_LDA{1, runId}.Yroc_train); title('ROC in Train for LDA Classificator for subj: ', num2str(subj), 'and run: ', num2str(run)])
%         xlabel('False Positive Rate'); ylabel('True Positive Rate')
%         subplot(212)
%         plot(score_tabs_LDA{1, run}.Xroc_test, score_tabs_LDA{1, runId}.Yroc_test); title('ROC in Test for LDA Classificator for subj: ', num2str(subj), 'and run: ', num2str(run)])
%         xlabel('False Positive Rate'); ylabel('True Positive Rate')
%         pause()
    end
    
%% Mean of the results computed for the subject subj
score_tabs_RF = meanResults(score_tabs_RF, nruns);
score_tabs_LDA = meanResults(score_tabs_LDA, nruns);
score_tabs_QDA = meanResults(score_tabs_QDA, nruns);
score_tabs_SVM = meanResults(score_tabs_SVM, nruns);

if threshold == 0
    subj_score_RF{1, sId} = score_tabs_RF;
    subj_score_LDA{1, sId} = score_tabs_LDA;
    subj_score_QDA{1, sId} = score_tabs_QDA;
    subj_score_SVM{1, sId} = score_tabs_SVM;
else
    subj_score_fisher_RF{1, sId} = score_tabs_RF;
    subj_score_fisher_LDA{1, sId} = score_tabs_LDA;
    subj_score_fisher_QDA{1, sId} = score_tabs_QDA;
    subj_score_fisher_SVM{1, sId} = score_tabs_SVM;
end %if/else

end %for subj

%% Save the scores of all the classificators in the Results Directory
name_file=fullfile(datapath_results,'ClassificationResults.mat');
if threshold == 0
    save(name_file, 'subj_score_RF', 'subj_score_LDA','subj_score_QDA','subj_score_SVM')
else
    save(name_file, 'subj_score_fisher_RF', 'subj_score_fisher_LDA', 'subj_score_fisher_QDA','subj_score_fisher_SVM')
end %if/else

%The results are located in subj_score_CLASS = 1x5cell, which contains the
%classification scores for the CLASS type of classifier, for all the 5 subjects (and so, for all the 5
%classifiers).
%Every element of subj_score is a 1x11 cell where:
%     - the elements from 1 to 10: are the scores of each iteration (Leave
%     one out type) of the classifier --> so considering that specific
%     run as test set and all the other runs as training set;
%     - the 11th element: contains the Mean Scores, computed on all the
%     first 10 iterations of the classifier;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% VISUALIZATION OF THE POPULATION

%% Visualizing the Grand Averages for the population
figure()
for sId=1:nsubjects
    plot(t_err_vis, mean(GA_err_vis_matrix{sId},2), 'Linewidth', 1)
    hold on
    if spatial_filter=="CAR"
        xlim([0 1])
    else
        xlim([0.2 1])
    end
    grid on
    xlabel('time [s]')
    ylabel('Amplitude [\muV]')
end

corr_sig=mean(GA_corr_vis_matrix{1},2);
for sId=2:nsubjects
    media=mean(GA_err_vis_matrix{sId},2);
    corr_sig=horzcat(corr_sig, media);
end
corr_baseline=mean(corr_sig,2);
plot(t_err_vis,corr_baseline, 'k--');
xline(0.2, 'm--', 'Linewidth', 1);
warning('OFF');
legend('S1','S2','S3','S4','S5','Corr baseline')
title('Grand Average for population')

%% Visualizing the Grand Averages for the population
figure()
for sId=1:nsubjects
    plot(t_err_vis, mean(GA_err1_vis_matrix{sId},2), 'Linewidth', 1)
    hold on
    plot(t_err_vis, mean(GA_err2_vis_matrix{sId},2), 'Linewidth', 1)
    if spatial_filter=="CAR"
        xlim([0 1])
    else
        xlim([0.2 1])
    end
    grid on
    xlabel('time [s]')
    ylabel('Amplitude [\muV]')
end

plot(t_err_vis,corr_baseline, 'k--');
xline(0.2, 'm--', 'Linewidth', 1);
warning('OFF');
legend('S1 Err1', 'S1 Err2','S2 Err1', 'S2 Err2' ,'S3 Err1', 'S3 Err2','S4 Err1', 'S4 Err2', 'S5 Err1', 'S5 Err2', 'Corr baseline')
title('Grand Average for population Err1/Err2')




%%      FUNCTION IMPLEMENTED

%% Functuon conc_project
%Concatenating function
function [s, TYP, DUR, POS, SampleRate, Mk, Rk]=conc_project(gdf_files, datapath, Modality)
% This function concatenates the signal s and the vectors TYP, DUR and POS
% of all GDF files provided in the struct gdf_files. 'datapath' is the
% path of the directory where the files are stored.
% Modality tells if to concatenate:
% - 'Offline' --> only the offline files
% - 'Online' ---> only the online files
% - 'All' --> online and offline files
% OUTPUT: the matrix s [samples x channels] with all the signals
% concatenated and the vectores TYP, DUR, POS obtained by the concatenation
% of the single signal's corresponding vectors. 
% SampleRate is the sample rate of the signal. 
% Mk is a vector [1 x total samples] with 1 if the samples belog to an
% online file, or 0 if they belong to an offline file.
% Rk is the [1 x total samples] vector of the runs, it tells to which run
% every sample of the signal s belongs. 

s=[];
POS=[];
DUR=[];
TYP=[];
Mk=[]; 
Rk=[];
idx_online=[];
idx_offline=[];

nfiles=length(gdf_files);
for fid=1:nfiles
    cfilename=fullfile(datapath, gdf_files(fid).name);
    
    if( contains(cfilename, 'offline') == true)
        idx_offline=[idx_offline, fid];
    elseif ( contains(cfilename, 'online') == true )
        idx_online=[idx_online, fid];
    else
        error(['Unknown modality for run: ' cfilename]);
    end
end
    
if Modality=='Offline'
    gdf_files(idx_online)=[];
elseif Modality=='Online'
    gdf_files(idx_offline)=[];
end

nfiles=length(gdf_files);

r=1;
for fid=1:nfiles
    cfilename=fullfile(datapath, gdf_files(fid).name);
    [cs,ch]=sload(cfilename);
    cevents=ch.EVENT;
    
    TYP=vertcat(TYP, cevents.TYP);
    % Adding the number of previous samples to the position values
    ns=length(s); %number of previous samples
    csamples=size(cs,1);
    cpositions=cevents.POS+ns;
    POS=vertcat(POS, cpositions);
    
    % For some reason, in the analysed GDF files the duration of every
    % event is labelled as 0. So, to overcome that, we directly calculate
    % the duration of every event in this way. 
    
    cDUR=[];
    for dId=2:length(cevents.POS)
        cDUR(dId-1)=cevents.POS(dId)-cevents.POS(dId-1);
    end
    ultima=csamples-cevents.POS(dId);
    cDUR=[cDUR,ultima];
    DUR=[DUR,cDUR];
    
    if( contains(cfilename, 'offline') == true)
        Mk(ns+1:ns+csamples)=0;
    elseif ( contains(cfilename, 'online') == true )
        Mk(ns+1:ns+csamples)=1;
    else
        error(['Unknown modality for run: ' cfilename]);
    end
    
    Rk(ns+1:ns+csamples)=r;
    
    s=vertcat(s,cs);
    SampleRate=ch.SampleRate;
    r=r+1;
    
end
end

%% Function extract_runs
function M=extract_runs(FeatureData, trial_runs, crun)
% OUTPUT = Matrix M of dimensions [n_extracted_samples x channels x trials]
% with all the data for a selected run.
% INPUT: 
% - FeatureData = total matrix of features
% - trial_runs = vector of dimension [1 x ntrials] with the reference of
% every trial's run.
% - crun = selected run.

ctrials=find(trial_runs == crun);
M=FeatureData(:,:,ctrials);

end


%% Function Labelling_project

function [RaceStart, RaceEnd, TurnLeftSession, TurnRightSession, LightSession, StraightSession, CommLeft, CommRight, comm_start, idx_comm]=labelling_project(nsamples, event)
% Fuction that, given as inputs:
% - nsamples: number of samples of signal s
% - event = struct with all the events
% The function gives as output the following label vectors:
% - RaceStart = vector in which all non-zero samples correspond to the
% race-start event
% - RaceEnd = vector in which all non-zero samples correspond to the
% race end event
% - TurnLeftSection= vector [1 x nsamples] with 201 in the samples 
% belonging to the turn left section and 0 elsewhere
% - TurnRightSection= vector [1 x nsamples] with 203 in the samples 
% belonging to the turn right section and 0 elsewhere
% - LightSection= vector [1 x nsamples] with 202 in the samples 
% belonging to the light section and 0 elsewhere
% - StraightSection= vector [1 x nsamples] with 204 in the samples 
% belonging to the stay straight section and 0 elsewhere
% - CommLeft = vector [1 x nsamples] with 101 in the samples 
% where the user commands to turn left and 0 elsewhere
% - CommRight = vector [1 x nsamples] with 103 in the samples 
% where the user commands to turn right and 0 elsewhere
% - comm_start= starting positions (in samples) corresponding to the 
% samples for which a command has been sent
% - idx_comm = indexes of EVENTS where a useful command was sent ('useful'
% means that the command has been sent during the race, while the commands
% sent after the end or before the beginning should not be considered).

TurnLeftSession=zeros(1,nsamples);
TurnRightSession=zeros(1, nsamples);
LightSession=zeros(1,nsamples);
StraightSession=zeros(1,nsamples);
CommLeft=zeros(1,nsamples);
CommRight=zeros(1,nsamples);
RaceStart=zeros(1,nsamples);
RaceEnd=zeros(1,nsamples);
comm_start=[];
idx_comm=[];

nevents=length(event.TYP);

for ii=1:nevents
    c_typ=event.TYP(ii);
    c_pos=event.POS(ii);
    c_dur=event.DUR(ii);

    %Turn Left Session
    if c_typ==201
        TurnLeftSession(c_pos:c_pos+c_dur-1)=c_typ;
  
    %Turn Right Session
    elseif c_typ==203
        TurnRightSession(c_pos:c_pos+c_dur-1)=c_typ;
        
    %LightSession
    elseif  c_typ==202
        LightSession(c_pos:c_pos+c_dur-1)=c_typ;

    %StraigthSession
    elseif c_typ==204
       StraightSession(c_pos:c_pos+c_dur-1)=c_typ;
    
    % User Sent command TURN LEFT
    elseif c_typ==101
        CommLeft(c_pos:c_pos+c_dur-1)=c_typ;
        % Here we're cleaning the information regarding the commands sent
        % after the end of one race and before the start of the following one.
        % We're cleaning the events that have as a previous event one
        % marked as 8800 (end of the race) or have a previous event that is
        % a command but it is not present in comm_start (which means that
        % command itself came after the end of the race). 
        % Sometimes the label 800 for the start of a race is not provided:
        % nevertheless, to start the race the avatar enters at least one
        % section, so the following code works also without the 800 label.
        if ii~=1 && event.TYP(ii-1)~=8800 
            if ((event.TYP(ii-1)==101 || event.TYP(ii-1)==103) && not(ismember(event.POS(ii-1), comm_start)))
                comm_start=comm_start;
            else
                comm_start=[comm_start,c_pos];
                idx_comm=[idx_comm, ii];
            end
        end 
    
    % User Sent command TURN RIGHT
    elseif c_typ==103
        CommRight(c_pos:c_pos+c_dur-1)=c_typ; 
        if ii~=1 && event.TYP(ii-1)~=8800 
            if ((event.TYP(ii-1)==101 || event.TYP(ii-1)==103) && not(ismember(event.POS(ii-1), comm_start)))
                comm_start=comm_start;
            else
                comm_start=[comm_start,c_pos];
                idx_comm=[idx_comm, ii];
            end
        end 
    
    %Race Start
    elseif c_typ==800
        RaceStart(c_pos:c_pos+c_dur-1)=c_typ; 
    
    %Race End
    elseif c_typ==8800
        RaceEnd(c_pos:c_pos+c_dur-1)=c_typ; 
    
    else %Indexes that are not present
        disp('There is an index that does not fall in any classification')
        disp(['The index is: ' num2str(c_typ)])
    end

end
end

%% Function extractLabels2 
%that labels the trials as correct or error trials
function labels = extractLabels2(features, eventi)

%This function extracts the label of the data from :
% features --> a struct of the feature obtained in the first part of the code 
%              (in particular, the matrix FeaturesData of the signal values after every
%              decision event (labelled as 101:user sent command turn left or 103:user sent command turn right)
%              in windows that starts 200 ms after that event and ends after 100 ms the
%              same event
% eventi -->   a struct with the events informtion of the patient (type,
%              duration an beginning of each labelled event) 
%
% and returns a vector labels --> that contains the class label (1 = error
%                                 trial, 0 = correct trial) of each identified trial(event of type 101 or
%                                 103, so a decisional event)

%same as before but I distinguish the two types of error:
% - LIGHT/STRAIGHT ERROR TRIAL: when a comand 'turn right' or 'turn left' is
% found after the avatar enters in a section in which he is not supposed to
% do anything --> -1 
% - CURVE ERROR TRIAL: when in a section 'turn right' the user turns left (and then corrects
%himself, by turning right) and viceversa --> 1
% - CORRECT TRIAL --> 0

%in this way abs(label error ) = 1

labels = nan(1, size(features.FeaturesData , 3));
i = 0; last_sector = 0;

for ev = 1:length(eventi.DUR)
    
    if startsWith(num2str(eventi.TYP(ev)), '2')
        last_sector = eventi.TYP(ev); %last sector/zone in which the avatar enters
    end %if
    
    if eventi.TYP(ev) == 101
        if i + 1 <= length(features.comm_start)
            if (eventi.POS(ev) == features.comm_start(i+1))
                i = i + 1;
                if last_sector ~= 201
                    if last_sector == 203
                        labels(i) = 1;
                    else
                        labels(i) = -1;
                    end %if/else
                else
                    labels(i) = 0;
                end %if/else
            end %if
        end %if
    end %if
        
    if eventi.TYP(ev) == 103
        if i + 1 <= length(features.comm_start)
            if (eventi.POS(ev) == features.comm_start(i+1))
                i = i + 1;
                if last_sector ~= 203
                    if last_sector == 201
                        labels(i) = 1;
                    else 
                        labels(i) = -1;
                    end %if/else
                else
                    labels(i) = 0;
                end %if/else
            end %if
        end %if
    end %if
    
end %for
%labels(features.indexes_todel) = []; 
end %function

%% Function CCA_filtering
function [Wx, U, Epochs_filtered]=CCA_filtering(FeatureData, FeatureTrain, eventi, comm_start, trial_runs, runs_test)
% This function computes the filtering matrix Wx
% INPUT: 
% - Matrix FeatureData [samples x channels x trials] ---> needed for the
% labels
% - Matrix FeatureTrain [samples x channels x trials_belonging_to_the_trainset]
% - struct eventi
% - comm_start (vector with the sample position of the start of every trial)
% - trial runs (vector in which every trial is assigned to the
% corresponding run)
% - runs_test (runs which constitute the test set)
% OUTPUT:
% - Matrix Wx (matrix to use as a spatial filter)
% - Matrix U (U=(X-mean(X))*Wx)
% - Matrix Epochs_filtered (filtering applied to FeatureTrain) 

nchannels=size(FeatureTrain, 2);
nsamples_epoch=size(FeatureTrain,1);
ntrials=size(FeatureTrain,3);

% EpochData has dimensions [nchannels x nsamples x ntrials];
EpochData=permute(FeatureData, [2 1 3]);

% Setting the struct to use inside extractLabels2;
mCCA.FeaturesData=EpochData;
mCCA.comm_start=comm_start;
etichette = extractLabels2(mCCA, eventi);
etichette=etichette(trial_runs~=runs_test); %considering only the labels belonging to the trainset
% -1 o 1 when the error should be present

EpochData=permute(FeatureTrain, [2 1 3]);

%Selecting the trials where the error is present
% X1: matrix containing the trials with the Error Potential
x1=EpochData(:,:,etichette~=0);
ntrials_err=size(x1,3);
X1=nan(nchannels, nsamples_epoch*ntrials_err);
for cId=1:nchannels
    k=0;
    for nId=1:ntrials_err
        X1(cId,k*nsamples_epoch+1:nId*nsamples_epoch)=x1(cId,:,nId);
        k=k+1;
    end
end

% X2: matrix containing the trials without the Error Potential
x2=EpochData(:,:, etichette==0);
ntrials_normal=size(x2,3);
X2=nan(nchannels, nsamples_epoch*ntrials_normal);
for cId=1:nchannels
    k=0;
    for nId=1:ntrials_normal
        X2(cId,k*nsamples_epoch+1:nId*nsamples_epoch)=x2(cId,:,nId);
        k=k+1;
    end
end

%Computing the final matrix X
X=[X1,X2]';

%Obtaining Y1 for the trials with the error potential
y1=mean(x1, 3); % y is a matrix [channels x samples]
Y1=nan(nchannels, nsamples_epoch*ntrials_err);
for cId=1:nchannels
    k=0;
    for nId=1:ntrials_err
        Y1(cId,k*nsamples_epoch+1:nId*nsamples_epoch)=y1(cId,:);
        k=k+1;
    end
end

%Obtaining Y2 for the trials without the error potential
y2=mean(x2,3); % y is a matrix [channels x samples]
Y2=nan(nchannels, nsamples_epoch*ntrials_normal);
for cId=1:nchannels
    k=0;
    for nId=1:ntrials_normal
        Y2(cId,k*nsamples_epoch+1:nId*nsamples_epoch)=y2(cId,:);
        k=k+1;
    end
end

%Computing the matrix Y
Y=[Y1,Y2]';

% Now we are applying canoncorr to find Wx
[Wx,Wy, R, U, V, stats]=canoncorr(X,Y);

% Now I've got two alterntives:
% - applying the filter to the data in EpochData
% - applying the filter to X, obtainining U, and from U
% reconstructing the original matrix
% ---> SPOILER: the two procedures lead to the same results!

% First option
EEG_data=nan(nchannels, nsamples_epoch*ntrials);
for cId=1:nchannels
    k=0;
    for trId=1:ntrials
        EEG_data(cId,k*nsamples_epoch+1:trId*nsamples_epoch)=EpochData(cId,:,trId);
        k=k+1;
    end
end

% Applying the spatial filter:
EEG_filtered=Wx'*EEG_data;

% I will now rebuild the matrix with the structure of
% [nsamples x nchannels x ntrials]
Epochs_filtered=nan(nsamples_epoch, nchannels, ntrials);
for cId=1:nchannels
    k=0;
    for nId=1:ntrials
        Epochs_filtered(:, cId, nId)=EEG_filtered(cId, k*nsamples_epoch+1:nId*nsamples_epoch);
        k=k+1;
    end
end

end



%% Function Feature Selection
% created in order to disciminate between the feature values of the two
% classes
function [featuresMatrixClass1, featuresMatrixClass2] = features_selection(X, y)
% it returns:
% featuresMatrixClass1 : the features matrix only for the error trials
% featuresMatrixClass2 : the features matrix only for the correct trials

% Initializing the variables
nfeatures=24;
nsubjects=size(X, 1);
nruns=size(X, 2);
% Feature matrices for 2 classes
featuresMatrixClass1=cell(nsubjects, nruns);
featuresMatrixClass2=cell(nsubjects, nruns);
% separate X into classes
for subj=1:size(X, 1)
    for run=1:nruns
        curr_run=X{subj, run};
        curr_labels=y{subj, run};

        % Creating labels matrix to have size(run)==size(labels)
        labels_matrix=nan(nfeatures, size(curr_labels, 2));
        for i=1:nfeatures
            labels_matrix(i, :)=abs(curr_labels);
        end

        % data (features) for error potential
        class1=curr_run.*labels_matrix;
        % data (features) for correct trials
        class2=curr_run.*abs((labels_matrix-1));
        % Removing the 0 columns (aka the features belonging to another
        % class)
        class1( :, ~any(class1,1))=[];
        class2( :, ~any(class2,1))=[];
       
        featuresMatrixClass1{subj, run} = class1;
        featuresMatrixClass2{subj, run} = class2;
%         % Check if the first run 
%           if run==1
%               subj_class1=class1;
%               subj_class2=class2;
% 
%           else
%               subj_class1 = cat(2, subj_class1, class1);
%               subj_class2 = cat(2, subj_class2, class2);
%          end
    end
    % Initialize (or expand) the feature matrix
%     if subj==1
%         featuresMatrixClass1{subj}=subj_class1;
%         featuresMatrixClass2{subj}=subj_class2;
%     else
%         featuresMatrixClass1{subj} = cat(2, featuresMatrixClass1{subj}, subj_class1);
%         featuresMatrixClass2{subj} = cat(2, featuresMatrixClass2{subj}, subj_class2);
%     end
end %subj
end %function

%% Function modelResults
function [TP, TN, FN, FP, error] = modelResults(y, predicted_labels)
% created in order to compute automatically all the result parameters
% as TP, TN, FN, FP also if using and comparing more classificators

error = 0;
TP = 0; TN = 0; FN =0 ; FP = 0; 
        
for i = 1:length(predicted_labels)
    if predicted_labels(i) ~= y(i)
        error = error + 1;
        if predicted_labels(i) == 1
            FN = FN + 1;
        else
            FP = FP + 1;
        end %if/else
    else
        if predicted_labels(i) == 1
            TN = TN + 1;
        else
            TP = TP + 1;
        end %if/else
    end %if/else (outside)
end %for
error = error / length(y);

end %function 


%% Function resultStruct
function score_tabs = resultStruct(score_tabs, run, train_error, test_error, TP_train, TN_train, FN_train, FP_train, TP_test, TN_test, FP_test, FN_test, y_train, y_test, score_train, score_test)
%On Train
score_tabs{1, run}.TRAIN_ERROR = train_error;
score_tabs{1, run}.TP_train = TP_train;
score_tabs{1, run}.TN_train = TN_train;
score_tabs{1, run}.FP_train = FP_train;
score_tabs{1, run}.FN_train = FN_train; 
score_tabs{1, run}.Accuracy_Train = ((TP_train + TN_train) / (TP_train + TN_train + FP_train + FN_train));
score_tabs{1, run}.TPR_train = ( TP_train /(TP_train + FN_train) );
score_tabs{1, run}.FPR_train = ( FP_train / (FP_train + TN_train) );

[Xroc_train, Yroc_train, Troc_train, AUC_train] = perfcurve(y_train, score_train(:,1), 0);

score_tabs{1, run}.AUC_train = AUC_train;
score_tabs{1, run}.Xroc_train = Xroc_train;
score_tabs{1, run}.Yroc_train = Yroc_train;

% On Test
score_tabs{1, run}.TEST_ERROR = test_error;
score_tabs{1, run}.TP_test = TP_test;
score_tabs{1, run}.TN_test = TN_test;
score_tabs{1, run}.FP_test = FP_test;
score_tabs{1, run}.FN_test = FN_test; 
score_tabs{1, run}.Accuracy_Test = ((TP_test + TN_test) / (TP_test + TN_test + FP_test + FN_test));
score_tabs{1, run}.TPR_Test = ( TP_test /(TP_test + FN_test) );
score_tabs{1, run}.FPR_Test = ( FP_test / (FP_test + TN_test) );

[Xroc_test, Yroc_test, T_roc_test, AUC_test] = perfcurve(y_test, score_test(:,1), 0);

score_tabs{1, run}.AUC_test = AUC_test;
score_tabs{1, run}.Xroc_test = Xroc_test;
score_tabs{1, run}.Yroc_test = Yroc_test;
end %function


%% Function meanResults
function score_tabs = meanResults(score_tabs, nruns)
% modifies the score_tabs struct, adding for every subject a 11th element
% which only contains the means of AUCs, Accuracies, FPRSs for both train
% and test

aucs_test = []; aucs_train = []; 
acc_test = []; acc_train = []; fprs_test = []; fprs_train = [];

for run = 1:nruns
        aucs_test = [aucs_test score_tabs{1, run}.AUC_test];
        aucs_train = [aucs_train score_tabs{1, run}.AUC_train];
        acc_test = [acc_test score_tabs{1, run}.Accuracy_Test];
        acc_train = [acc_train score_tabs{1, run}.Accuracy_Train];
        fprs_test = [fprs_test score_tabs{1, run}.FPR_Test];
        fprs_train = [fprs_train score_tabs{1, run}.FPR_train];
end %for

avg_results.Mean_AUC_test = mean(aucs_test);
avg_results.Mean_AUC_train = mean(aucs_train);
avg_results.Mean_Accuracy_Test = mean(acc_test);
avg_results.Mean_Accuracy_Train = mean(acc_train);
avg_results.Max_Accuracy_Test = max(acc_test);
avg_results.Mean_FPR_Test = mean(fprs_test);
avg_results.Mean_FPR_Train = mean(fprs_train);

score_tabs{1, nruns + 1} = avg_results;

end %function