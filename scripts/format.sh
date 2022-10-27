#! /bin/sh
clang-format -i **/*.cpp > /dev/null
clang-format -i **/*.h > /dev/null
echo "Formatted all project code!"
