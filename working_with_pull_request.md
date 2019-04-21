# Quick start for working with a pull request from the teminal

change working directory
```
cd YourWorkingDirectory
```
clone project
```
git clone https://github.com/WomenInDataScience-Seattle/Machine_Learning_Projects.git
```
type username and password if needed

then, create a test branch
```
git branch YourBranchName
```
checkout branch, meaning that you will be working on that branch
```
git checkout YourBranchName
```
on your branch, work on the changes
after editing your file, commit it and push to the test branch
```
git add FileYouEdited
git commit -m "removing blank lines"
git push origin YourBranchName
```
go to your branch, send a pull-request, and request for review (please add @ddong63 as a default reviewer) 

after you get approved, merge the branch (resolve confilts if needed), add informative commit message, and delete the branch

**A tutorial can be found here (5-min)**
https://www.youtube.com/watch?v=FQsBmnZvBdc
