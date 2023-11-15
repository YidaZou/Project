for file in *-1
do
  mv "$file" "${file/-1/-1%perturbed}"
done