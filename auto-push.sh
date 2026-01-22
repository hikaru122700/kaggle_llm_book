#!/bin/bash

while true; do
    timestamp=$(date '+%H:%M:%S')

    # 変更があるかチェック
    if git diff --quiet && git diff --cached --quiet; then
        echo "[$timestamp] No changes to push"
    else
        # add & commit & push
        git add -A
        if git commit -m "Auto-commit at $timestamp" --quiet 2>/dev/null; then
            if git push --quiet 2>/dev/null; then
                echo "[$timestamp] Pushed successfully"
            else
                echo "[$timestamp] Push failed"
            fi
        else
            echo "[$timestamp] Nothing to commit"
        fi
    fi

    sleep 10
done
