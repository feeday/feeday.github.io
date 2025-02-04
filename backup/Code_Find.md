## Windows 查找
```
Get-ChildItem -Recurse -File | Select-String -Pattern "56"
```
## Bash 查找
搜索匹配当前目录下所有文件名或文件里的内容打印显示行号
```
grep -rn "Money" *
```
搜索多个文件并查找匹配文本在哪些文件中：
```
grep -l "Money" file1 file2 file3...
```
查找 formatting.php 文件内 length’, 55 替换 length’, 56 :

```
sed -i s/"length', 55"/"length', 56"/g `grep "length', 55" -rl --include="formatting.php" ./`
```
