function [true] = my_plot3d(X,Y,view0,pcolor,size)
%view=[0,0,1280,800];   'MarkerSize', 8
 set(gcf,'position',view0);
    % subplot(1,2,1)
    hold on
    % axis equal
%         title('Points Cloud','fontsize',14)
    if pcolor==1
    plot3(X(:,1),X(:,2),X(:,3),'g.','MarkerSize', size)
    end
    if pcolor==2
    plot3(X(:,1),X(:,2),X(:,3),'r.','MarkerSize', size)
    end
 
%     view(-30,-120);
    view(2); %  view(-45,45);

    %% plot of the output triangulation
    % figure(3)
    % subplot(1,2,2)
    hold on
%         title('Points Cloud','fontsize',14)
    % axis equal
    plot3(Y(:,1),Y(:,2),Y(:,3),'b.','MarkerSize', size)
    % trisurf(t,Y(:,1),Y(:,2),Y(:,3),'facecolor','c','edgecolor','b')%plot della superficie trattata
%     view(-30,-120);%horse (90,0) bunny view(-30,-120)
    axis off;
    axis( 'equal');
    view(2); %  view(-45,45);
true=1;
end