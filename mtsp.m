% author: zzilch
% email: zz@zilch.zone
% date: 2019/11/26
function mtsp(file_id,nring)
% file_id: the data file id '48'/'29'/'51'/'76'/'101'/'225'/'561'
% nring: the traveler number 1,2,3,...

% file_id = '561';
files = containers.Map;
files('48')='att48';
files('29')='bayg29';
files('51')='eil51';
files('76')='eil76';
files('101')='eil101';
files('225')='tsp225';
files('561')='pa561';
tspdata = load(['data/' files(file_id) '.tsp' '.txt']);
tsptour = load(['data/' files(file_id) '.tour' '.txt']);

format;
% Data info
nsize = size(tspdata); % data size
ncity = nsize(1); % city number
cities = tspdata(:, 2:3); % (x,y) of the cities
max_bound = max(cities); % max bound of all cities
min_bound = min(cities); % min bound of all cities
center = (max_bound - min_bound) / 2 + min_bound; % centroid of all cities
min_dist = min(pdist(cities)); % min distance among all cities
th = 0.01*min_dist; % distance error tolerance

% SOM hyper-parameters
%nring = 1; % number of travelers/rings
nnode = ncity; % number of cities for each traveler
if nring>1
    nnode = ceil(ncity/nring);
end
beta = 0.1; % learning rate 1.0/0.3/0.1/0.03
decay = 0.998; % learning rate decay
gain = 10; % neighbor function gain
alpha = 0.03; % gain decay
eta = 0.2; % percent of neiborhoods 1.0/0.5/0.1/0.05/0.01

% Training
tstart = cputime;
som = som_init(nring,nnode,center,min_dist/2); 
som_old = som; % keep the old weights
niter = 0;
solution = zeros(nring, nnode);
while true
    % visulization
%     plot(cities(:, 1), cities(:, 2), 'ko', 'MarkerFaceColor', 'r');
%     hold on;
%     drawtours(som);
%     title([files(file_id) ' MTSP: ' num2str(nring) 'Traveler' ', Iter:' num2str(niter)])
% %     if mod(niter,10)==0
% %         saveas(gcf,join(["output/" files(file_id) '_' num2str(nring) '_' num2str(niter) '.png']));
% %     end
%     pause(0.0001);
%     hold off;
    
    perms = randperm(ncity); % shuffle the data
    inhibit = zeros(nring,nnode); % banned cites
    for i = 1:ncity
        city = cities(perms(i), :);
        % caculate the costs for all travelers
        % https://www.sciencedirect.com/science/article/abs/pii/S0305054898000690
        costs = tourcosts(som);
        costs_avg = mean(costs);
        costs = 1+(costs-costs_avg)/costs_avg;
        
        % select the winner traveler and node
        [win_ring,win_node] = select_winner(som,city,costs,inhibit);
        solution(win_ring,win_node) = perms(i);
        inhibit(win_ring,win_node) = 1;
        
        % update the weights of winner traveler
        som = som_adapt(som,win_ring,win_node,city,beta,gain,eta);
        
        % calculate the difference with GT
        err = sqrt((city(1) - som(win_ring, win_node, 1))^2 + (city(2) - som(win_ring, win_node, 2))^2);
        if err < th
            som(win_ring, win_node, 1) = city(1);
            som(win_ring, win_node, 2) = city(2);
        end
    end
    
    if som == som_old
        break;
    end
    
    % update learning rate and gain
    som_old = som;
    beta = beta * decay;
    gain = (1 - alpha) * gain;
    niter = niter + 1;
end
t = cputime - tstart;

% opt costs
gt = [tsptour; tsptour(1)];
gt_tour = zeros(ncity+1,2);
for n = 1:ncity+1
    gt_tour(n, :) = cities(gt(n), :);
end
gt_cost = tourcost(gt_tour);
cost = sum(tourcosts(som));
cost_diff = cost-gt_cost;

% visualization
figure;
sgtitle([files(file_id) 'MTSP: ' num2str(nring) 'Traveler']);
set (gcf,'Position',[100,100,1000,400])
subplot(1,2,1);
plot(cities(:, 1), cities(:, 2), 'ko', 'MarkerFaceColor', 'r');
hold on;
drawtours(som);
hold off;
t = ['Iter:' num2str(niter) ', Costs:' num2str(cost) ', Time:' num2str(t) 's'];
title(t);
subplot(1,2,2);
plot(gt_tour(:, 1), gt_tour(:, 2), 'sb-');
t = [ 'OptCosts: ' num2str(gt_cost) ', Error:' num2str(cost_diff)];
title(t);
saveas(gcf,join(["output/" files(file_id) '_' num2str(nring) '_' 'result' '.png']));

end

function drawtours(tours)
% tours: som weights or paths of city coordinates for all travelers
    for i = 1:size(tours,1)
        plot(squeeze([tours(i, :, 1) tours(i,1,1)]),squeeze([tours(i, :, 2) tours(i,1,2)]), '.-');
    end 
end

function [win_ring,wind_node] = select_winner(som,city,costs,inhibit)
% som: som weights
% city: target city coordinates
% costs: traveler costs calculated by formula in https://www.sciencedirect.com/science/article/abs/pii/S0305054898000690
% inhibt: banned cities
    nring = size(som,1);
    nnode = size(som,2);
    dists = ones(nring,nnode)*inf;
    for i = 1:nring
        for j = 1:nnode
            if inhibit(i,j) == 1
                dists(i,j) = inf;
            else
                dists(i,j) = costs(i)*(city(1) - som(i,j,1))^2 + (city(2) - som(i,j,2))^2;
            end
        end
    end
    [min_dist,win_ring]=min(dists,[],1);
    [~,wind_node]=min(min_dist);
    win_ring = win_ring(wind_node);
end

function som = som_init(nring,nnode,center,r)
% nring: travelers number
% nnode: city nubmer for each traveler
% center: initial center
% r: initial radius
    som = ones(nring,nnode,2);
    for i = 1:nring
        r1 = r;
        theta1 = 2*i*pi/nring;
        for node = 1:nnode
            r2 = r*sin(theta1/2);
            theta2 = 2*node*pi/nring;
            som(i,:,1) = center(1)+r1*cos(theta1)+r2*cos(theta2);
            som(i,:,2) = center(2)+r2*sin(theta1)+r2*cos(theta2);
        end
    end
end

function som = som_adapt(som,win_ring,win_node,city,beta,gain,eta)
% som: som weights
% win_ring : winner traveler
% win_node : winner node of winner traveler
% city: target city
% beta: learning rate
% gain: nerbor function gain
% eta: percent of nerborhoods
    nnode = size(som,2);
    for node = 1:nnode
        d = min(abs(node - win_node), nnode - abs(node - win_node));
        if d < eta * nnode
            f = exp(-d * d / (gain * gain));
            som(win_ring, node, 1) = som(win_ring, node, 1) + beta * f * (city(1) - som(win_ring, node, 1));
            som(win_ring, node, 2) = som(win_ring, node, 2) + beta * f * (city(2) - som(win_ring, node, 2));
        end
    end
end

function cost = tourcost(tour)
% tour: som weights or gt city coordinates of one traveler
    diff_w = [diff(tour);tour(end, :)-tour(1, :)];
    cost = sum(sqrt(sum(diff_w.*diff_w,2)));
end

function costs = tourcosts(tours)
% tours: som weights or gt city coordinates of all travelers
    nring = size(tours,1);
    costs = zeros(1,size(tours,1));
    for i = 1:nring
        tour = squeeze([tours(i,:,:) tours(i,1,:)]);
        costs(i) = tourcost(tour);
    end
end