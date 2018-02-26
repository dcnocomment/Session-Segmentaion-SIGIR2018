###need pre-trained embeddings
dict=$1
lowerad_term_vec=$2
awk -F'\t' 'BEGIN{size=5000; d_size=50;}ARGIND==1{d[$1]=$2}ARGIND==2{split($1,s," "); v[s[1]]=$1;}END{for(i=0;i<size;i++){if(i in d){if(d[i] in v){print v[d[i]]}else{res=i; for(j=0;j<d_size;j++){res = res " " rand()*2-1}; print res;}}else{res=i; for(j=0;j<d_size;j++){res = res " " rand()*2-1}; print res;}}}' $dict $lowerad_term_vec > embedding

awk -F' ' 'BEGIN{d_size=50}{res=$2; for(i=3;i<=d_size+1;i++){res=res " " $i}; print res}' embedding > init_embedding

rm embedding
