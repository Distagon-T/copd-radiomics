myjudge<-function(a,b){ #ÅĞ¶ÏÏàÍ¬Óë·ñ
  c<-0
for(i in 1:length(a)){
  if(a[i]==b[i])
  {c[i]<-0}
  else{c[i]<-1}
}
  return(c)
}