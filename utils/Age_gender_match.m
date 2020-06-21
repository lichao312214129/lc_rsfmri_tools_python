function Age_gender_match(year_gap,group1,group2)


%% ----------------------------------------------------------------------
% This function age and gender matches two groups +/- the year gap given.
% It can be used when the case group was recruited and the researcher wants
% to find out, which ages and genders are required to match the control
% group. The two groups are matched randomly for many times, so that you 
% can determine the 'priority' of each gender and age that is unmatched.

%% INPUTs:
%% year_gap: Maximum difference in years between case and control group

%% group1: Case group (N x 4). 
% The columns represent the age and gender as
% follows: [years, months, days, gender]. The gender should be given a
% numeric value. E.g. male = 0; female = 1. See example below if nargin == 1.

%% group2: Control group (N x 4).
% Structure identical to group1 input.

%% CONTACT
% Lukas Gerald Wiedemann, April 2018
% Department of Mechanical Engineering, University of Auckland, New Zealand
% lwie327@aucklanduni.ac.nz
% http://www.mdt.auckland.ac.nz/lukas-wiedemann/

%% -----------------------------------------------------------------------

if nargin == 0
    year_gap = 1;
elseif nargin == 1 % sample data
    %% example data (group1 = case; group2 = control)
    group1 = [8,6,21,0;10,3,25,0;5,9,27,1;12,0,22,0;10,0,16,0;6,5,16,0;7,3,12,1;9,6,1,0;10,6,26,1;10,11,3,1;10,11,3,1;5,5,24,0;9,0,30,0;9,0,30,0;9,7,30,0]; % [years, months, days, gender] (GENDER m = 0; f = 1)
    group2 = [14,0,25,1;11,10,12,1;14,5,2,0;9,8,5,1;7,7,2,1;7,5,10,0;7,0,7,0;8,9,7,0;8,9,7,0;13,4,12,0;13,4,12,0;6,7,15,0;9,10,17,0;9,10,17,0;10,9,8,1;8,6,13,0;10,3,9,1;10,3,9,1;7,10,15,0;11,8,13,1;9,11,22,0;12,1,4,1;8,1,30,0;8,1,30,0;8,6,16,0;6,6,0,0;6,4,28,1;11,5,25,1];
    
end

sub_priority = zeros(1,length(group1));
for priority = 1:100000 % run randomized matching multiple times to obtain the priority to match each individual of the case group

    group1_age = group1(:,1)+group1(:,2)/12+group1(:,3)/365; 
    group2_age = group2(:,1)+group2(:,2)/12+group2(:,3)/365; 
    group1_gender = group1(:,4);
    group2_gender = group2(:,4);

    group1_random_ind = randperm(length(group1_age));
    group2_random_ind = randperm(length(group2_age));

    Match_cases = [];
    Unmatch_cases = 1:length(group1_age);

    for i = group1_random_ind
        for ii = group2_random_ind
           if abs(group1_age(i) - group2_age(ii)) <= year_gap & group1_gender(i) == group2_gender(ii)
%                disp(['match found: ' num2str(i)])
               Match_cases(end+1) = i;
               Unmatch_cases(i) = NaN;
               group2_age(ii) = NaN; % subject can only match once
               break;
           end
        end
    end

    Match_cases = sort(Match_cases);
    Unmatch_cases(isnan(Unmatch_cases)) = [];
    
    sub_priority(Unmatch_cases) = sub_priority(Unmatch_cases) + 1;
    
end

sub_priority = sub_priority./1000; % Matching-priority of each individual in %

%% Display results:
disp('Matched and Unmatched cases: ')
disp('0: Subject in case group is perfectly matched; >0 = Subject is not matched in x% of all cases; 100: Subject never matched')
for matched = 1:length(sub_priority)
    disp(['Subject ID ' num2str(matched) ': ' num2str(sub_priority(matched))])
end
