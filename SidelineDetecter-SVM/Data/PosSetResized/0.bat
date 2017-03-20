dir /b>0SetName.txt
@echo off&setlocal enabledelayedexpansion
for %%i in (0SetName.txt) do (
	set n=0
	for /f "usebackq delims=" %%j in ("%%i") do (
                           set/a n+=1
	           if !n! gtr 2 echo,%%j>>temp
                )
               move /Y temp "%%i"
)