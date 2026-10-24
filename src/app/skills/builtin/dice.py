"""`roll_dice`（契约 §2.5-2）：`<count>d<sides>` 掷骰，`times` 1~10 次。"""

import random
import re

from pydantic import BaseModel

from app.skills.base import BaseSkill, ToolContext, ToolResult

# ^([1-9]|10)d([2-9]|[1-9][0-9]|100)$（契约 §2.5-2）
DICE_RE = re.compile(r"^([1-9]|10)d([2-9]|[1-9][0-9]|100)$")


class RollDiceArgs(BaseModel):
    dice: str = "1d6"
    times: int = 1


def _invalid(dice: str, times: int, message: str) -> ToolResult:
    return ToolResult(
        ok=False,
        error_type="invalid_dice",
        message=message,
        data={"dice": dice, "times": times},
    )


async def roll_dice(ctx: ToolContext, args: RollDiceArgs, *, rng: random.Random | None = None) -> ToolResult:
    match = DICE_RE.match(args.dice)
    if match is None:
        return _invalid(args.dice, args.times, f"非法骰子表达式：{args.dice}")
    if not 1 <= args.times <= 10:
        return _invalid(args.dice, args.times, f"掷骰次数须在 1~10：{args.times}")

    count, sides = int(match.group(1)), int(match.group(2))
    randomizer = rng or random.Random()  # 生产不固定种子；测试可注入
    rolls = [
        [randomizer.randint(1, sides) for _ in range(count)] for _ in range(args.times)
    ]
    return ToolResult(
        ok=True,
        data={
            "rolls": rolls,
            "total": sum(sum(group) for group in rolls),
            "dice": args.dice,
            "times": args.times,
        },
    )


def make_roll_dice_skill(*, rng: random.Random | None = None) -> BaseSkill:
    async def handler(ctx: ToolContext, args: RollDiceArgs) -> ToolResult:
        return await roll_dice(ctx, args, rng=rng)

    return BaseSkill(
        name="roll_dice",
        description="掷骰子并返回每个骰子的点数与总和",
        when_to_use="需要掷骰时使用",
        parameters=RollDiceArgs,
        handler=handler,
    )
