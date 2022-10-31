#! /bin/sh
for folder in benchmark include src test
do 
	# echo $folder
	clang-format -i $folder/*.cpp > /dev/null
	clang-format -i $folder/*.h > /dev/null
	# ls $folder
done

# clang-format -i **/*.cpp > /dev/null
# clang-format -i **/*.h > /dev/null
echo "Formatted all project code!"
