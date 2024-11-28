function [T, Yc] = Pc_proj(Yin,T1)
Yc=[];
for i=1: max(Yin)
     ct=find(Yin==i);
     if size(ct,1)==1
        Yc=[Yc;T1(ct,:)];
     else
        Yc=[Yc;mean(T1(ct,:))];
     end
end
T=Yc;