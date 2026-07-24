# 公司 GitLab 同步说明

本文说明如何把 /m2v_intern/tujiahang/Projects/lerobot 中的代码同步到公司 GitLab，同时保留现有 GitHub remote。

## 固定信息

~~~bash
CORP_URL='https://git.corp.kuaishou.com/embodied-brain/interns/embodied-manipulation/frank3.git'
CORP_BRANCH='agent/franka-async-client-server'
CORP_EMAIL='tujiahang@kuaishou.com'
~~~

本文中的命令都使用完整 URL，不需要执行 git remote add，因此不会修改或覆盖现有的 origin、myfork 等 GitHub remote。

## 第一次配置邮箱

在原仓库中执行一次：

~~~bash
cd /m2v_intern/tujiahang/Projects/lerobot
git config --local user.name 'tujiahang'
git config --local user.email "$CORP_EMAIL"
~~~

这只影响以后创建的 commit，不会修改已有 commit，也不会改变 GitHub remote。

检查配置：

~~~bash
git config --local --get user.name
git config --local --get user.email
~~~

## 每次修改代码后的流程

### 1. 在原仓库提交代码

~~~bash
cd /m2v_intern/tujiahang/Projects/lerobot
git status --short
git add <需要同步的文件>
git commit -m '描述本次修改'
~~~

提交前确认没有误加入测试数据或其他 LFS 文件：

~~~bash
git diff --cached --name-only
git lfs ls-files --name-only
~~~

提交邮箱检查：

~~~bash
git show -s --format='%H%n%an <%ae>%n%cn <%ce>%n%s' HEAD
~~~

作者邮箱和提交者邮箱都应为 $CORP_EMAIL。

### 2. 获取公司 GitLab 当前分支

这条命令不会新增或修改 GitHub remote，只更新本地的 refs/remotes/corp/...：

~~~bash
git fetch "$CORP_URL" \
  "refs/heads/$CORP_BRANCH:refs/remotes/corp/$CORP_BRANCH"
~~~

第一次同步时，创建一个专门用于公司 GitLab 的工作树：

~~~bash
git worktree add -b corp-sync \
  /tmp/lerobot-corp-sync \
  "refs/remotes/corp/$CORP_BRANCH"
~~~

以后不要在原来的 GitHub 工作树上直接改写公司分支历史；公司工作树固定使用：

~~~text
/tmp/lerobot-corp-sync
~~~

### 3. 将原仓库的新 commit 应用到公司工作树

记录上一次已经同步到公司的“源仓库 commit”。例如第一次同步可以使用原始版本：

~~~bash
SOURCE_BASE='6309cb60498cf912a0e85395d28e3d12de8fb63f'
~~~

导出从 SOURCE_BASE 到当前 HEAD 的 commit，并应用到公司工作树：

~~~bash
PATCH_FILE='/tmp/lerobot-corp-sync.patch'
git format-patch --binary --stdout "$SOURCE_BASE..HEAD" > "$PATCH_FILE"
git -C /tmp/lerobot-corp-sync am "$PATCH_FILE"
~~~

如果只想同步某一个 commit，也可以使用：

~~~bash
git format-patch --binary --stdout -1 HEAD > "$PATCH_FILE"
git -C /tmp/lerobot-corp-sync am "$PATCH_FILE"
~~~

同步成功后，把本次原仓库的 HEAD 记为下一次的 SOURCE_BASE：

~~~bash
git rev-parse HEAD
~~~

不要把 myfork/agent/franka-async-client-server 强制推送到 GitHub；公司工作树中的 commit 会生成新的 hash，这是正常的。

### 4. 检查公司工作树中的邮箱和 LFS

~~~bash
git -C /tmp/lerobot-corp-sync log -5 \
  --format='%h %an <%ae> | %cn <%ce> %s'
~~~

检查是否有 LFS 指针：

~~~bash
git -C /tmp/lerobot-corp-sync grep -l \
  --fixed-strings 'version https://git-lfs.github.com/spec/v1' \
  HEAD -- || echo 'no LFS pointers'
~~~

如果输出了文件路径，不能用 GIT_LFS_SKIP_PUSH=1 绕过。需要先上传对应 LFS 对象：

~~~bash
git -C /tmp/lerobot-corp-sync lfs push --all "$CORP_URL" HEAD
~~~

如果这些 LFS 文件不需要同步，应在公司工作树中将它们从索引移除，再提交一个正常的公司 commit：

~~~bash
git -C /tmp/lerobot-corp-sync rm -r --cached <LFS文件或目录>
git -C /tmp/lerobot-corp-sync commit -m 'chore: exclude local LFS artifacts'
~~~

.gitignore 只对未跟踪文件生效；已经提交过的 LFS 文件必须使用 git rm --cached 移出索引。

### 5. 推送到公司 GitLab

确认工作树干净后推送：

~~~bash
git -C /tmp/lerobot-corp-sync status --short
git -C /tmp/lerobot-corp-sync push "$CORP_URL" \
  "HEAD:refs/heads/$CORP_BRANCH"
~~~

正常情况下不要使用：

~~~bash
GIT_LFS_SKIP_PUSH=1
git push --force myfork ...
~~~

前者会导致公司 GitLab 缺少 LFS 对象，后者会改写 GitHub 分支。公司 GitLab 的服务端 push rule 仍会检查 LFS 对象，跳过本地 hook 不能绕过服务端检查。

## 当前无历史 snapshot 的特殊情况

如果公司仓库使用的是“不保留历史”的根 snapshot，那么第一次成功推送后，公司分支与 GitHub 分支没有共同祖先。之后必须继续使用 /tmp/lerobot-corp-sync，通过 git format-patch / git am 把新改动应用到公司分支，不能直接把原 GitHub 分支推过去。

如果希望保留完整上游历史，应让公司 GitLab 管理员先导入上游 Git 历史和 LFS 对象，或允许已导入的 GitHub commit 使用外部邮箱；不要把其他贡献者的作者信息改成自己的邮箱。

## 常见错误

### Commit 提交者邮箱不符合仓库设置规范

检查 author 和 committer：

~~~bash
git log HEAD --format='%h %ae %ce %s' | \
  awk '$2 != "tujiahang@kuaishou.com" || $3 != "tujiahang@kuaishou.com"'
~~~

### GitLab: LFS objects are missing

说明 commit 中仍有 LFS 指针，但对象没有上传。执行：

~~~bash
git lfs push --all "$CORP_URL" HEAD
~~~

或者移除不需要的 LFS 文件后重新提交。

### non-fast-forward

不要立即使用 --force。先确认公司 GitLab 分支是否已经有其他人的提交，以及本地是否基于最新的公司分支：

~~~bash
git fetch "$CORP_URL" \
  "refs/heads/$CORP_BRANCH:refs/remotes/corp/$CORP_BRANCH"
git log --oneline --decorate -10 "refs/remotes/corp/$CORP_BRANCH"
~~~
