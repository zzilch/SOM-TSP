% author: zzilch
% email: zz@zilch.zone
% date: 2019/11/26
function mtsp(file_id)
% file_id = '29';
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
nnode = ncity; % number of travelers/rings
beta = 0.1; % learning rate 1.0/0.3/0.1/0.03
decay = 0.998; % learning rate decay
gain = 10; % neighbor function gain
alpha = 0.03; % gain decay
eta = 0.01; % percent of neiborhoods 1.0/0.5/0.1/0.05/0.01

% Training
tstart = cputime;
som = som_init(nnode,center,min_dist/2);
som_old = som; % keep the old weights
niter = 0;
solution = zeros(nnode);
while true
    % visulization
%     plot(cities(:, 1), cities(:, 2), 'ko', 'MarkerFaceColor', 'r');
%     hold on;
%     drawtour(som);
%     title([files(file_id) ' TSP, Iter:' num2str(niter)])
% %     if mod(niter,10)==0
% %         saveas(gcf,join(["output/" files(file_id) '_' num2str(niter) '.png']));
% %     end
%     pause(0.0001);
%     hold off;
    perms = 1:ncity;
    perms = randperm(ncity); % shuffle the data
    inhibit = zeros(1,nnode); % banned cites
    for i = 1:ncity
        city = cities(perms(i), :);
        
        % select the winner node
        [min_dist,win_node] = select_winner(som,city,inhibit);
        solution(win_node) = perms(i);
        inhibit(win_node) = 1;
        
        % update weights
        som = som_adapt(som,win_node,city,beta,gain,eta);
        
        % calculate the difference with GT
        err = sqrt((city(1) - som(win_node, 1))^2 + (city(2) - som(win_node, 2))^2);
        if err < th
            som(win_node, 1) = city(1);
            som(win_node, 2) = city(2);
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
cost = sum(tourcost(som));
cost_diff = cost-gt_cost;

% visualization
figure;
sgtitle([files(file_id) ' TSP']);
set (gcf,'Position',[100,100,1000,400])
subplot(1,2,1);
plot(cities(:, 1), cities(:, 2), 'ko', 'MarkerFaceColor', 'r');
hold on;
drawtour(som);
hold off;
t = ['Iter:' num2str(niter) ', Costs:' num2str(cost) ', Time:' num2str(t) 's'];
title(t);
subplot(1,2,2);
plot(gt_tour(:, 1), gt_tour(:, 2), 'sb-');
t = [ 'OptCosts: ' num2str(gt_cost) ', Error:' num2str(cost_diff)];
title(t);
saveas(gcf,join(["output/" files(file_id) '_' 'result' '.png']));

end

function drawtour(tour)
% tour: som weights or paths of city coordinates
    plot([tour(:, 1); tour(1,1)],[tour(:, 2); tour(1,2)], '.-');
end

function [min_dist,wind_node] = select_winner(som,city,inhibit)
% som: som weights
% city: target city coordinates
% inhibt: banned cities
    nnode = size(som,1);
    dists = ones(1,nnode)*inf;
    for j = 1:nnode
        if inhibit(j) == 1
            dists(j) = inf;
        else
            dists(j) = (city(1) - som(j,1))^2 + (city(2) - som(j,2))^2;
        end
    end
    [min_dist,wind_node]=min(dists);
end

function som = som_init(nnode,center,r)
% nnode: city nubmer for each traveler
% center: initial center
% r: initial radius
    som = ones(nnode,2);
    for node = 1:nnode
        theta = 2*node*pi/nnode;
        som(node,1) = center(1)+r*cos(theta);
        som(node,2) = center(2)+r*sin(theta);
    end
end

function som = som_adapt(som,win_node,city,beta,gain,eta)
% som: som weights
% win_node : winner node of winner traveler
% city: target city
% beta: learning rate
% gain: nerbor function gain
% eta: percent of nerborhoods
    nnode = size(som,1);
    for node = 1:nnode
        d = min(abs(node - win_node), nnode - abs(node - win_node));
        if d < eta * nnode
            f = exp(-d * d / (gain * gain));
            som(node, 1) = som(node, 1) + beta * f * (city(1) - som(node, 1));
            som(node, 2) = som(node, 2) + beta * f * (city(2) - som(node, 2));
        end
    end
end

function cost = tourcost(tour)
% tour: som weights or gt city coordinates of one traveler
    diff_w = [diff(tour);tour(end, :)-tour(1, :)];
    cost = sum(sqrt(sum(diff_w.*diff_w,2)));
end