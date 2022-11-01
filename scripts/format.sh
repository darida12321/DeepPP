#! /bin/sh
for folder in benchmark include src test
do 
	echo formatting $folder...
	find $folder -name "*.h" -o -name "*.cpp" -exec clang-format -i {} ';'
	# clang-format -i $folder/*.cpp $folder/*.h > /dev/null
	# clang-format -i $folder/*.h > /dev/null
	# ls $folder
done

# clang-format -i **/*.cpp > /dev/null
# clang-format -i **/*.h > /dev/null
echo "Formatted all project code!"
