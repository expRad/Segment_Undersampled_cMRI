function m = createRadialMask(traj,diameter)

% function m = createRadialMask(s, t)
%     puts radial traj with complex values [-0.5;0.5] onto cartesian grid
%     returns a diameter x diameter matrix
%     traj -- s -- number of spokes
%     t -- points within each spoke

[s,t]=size(traj);

m=zeros(diameter);
for a=1:s
    for b=1:t
        m(round(diameter/2+1+real(traj(a,b))*2*(diameter/2-1)),round(diameter/2+1+imag(traj(a,b))*2*(diameter/2-1)))=1;
    end
end
