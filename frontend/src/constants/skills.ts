/**
 * 内置工具清单（与后端 `src/app/skills/builtin/__init__.py` 的 5 个工具保持一致）。
 *
 * 后端暂无 `GET /skills` 只读端点，因此前端先用常量维护；registry 变更时需同步本文件。
 */

export interface SkillMeta {
  /** 与后端 registry 完全一致的名称，写入 `allowed_skills`。 */
  name: string
  /** 徽标上的短标签。 */
  label: string
  /** 表单里的完整说明。 */
  description: string
  /** 使用限制/注意事项（展示在表单与 tooltip 中）。 */
  note?: string
}

export const SKILLS: SkillMeta[] = [
  {
    name: 'get_current_time',
    label: '时间',
    description: '查询当前日期与时间（可指定时区）。',
  },
  {
    name: 'roll_dice',
    label: '掷骰',
    description: '掷骰子并返回每个骰子的点数与总和。',
  },
  {
    name: 'web_search',
    label: '搜索',
    description: '联网检索最新信息并返回网页摘要。',
    note: '后端未配置搜索 key 时会自动不下发该工具。',
  },
  {
    name: 'set_reminder',
    label: '提醒',
    description: '创建一条待确认的提醒事项，需用户在提醒面板确认后生效。',
    note: '仅私聊有效；群聊工具上下文 session_id=0，不会创建提醒。',
  },
  {
    name: 'read_lore',
    label: '世界书',
    description: '主动检索该角色世界书中的设定片段。',
    note: '需先在「世界书」页为该角色导入设定文档。',
  },
]

export const SKILL_MAP: Record<string, SkillMeta> = Object.fromEntries(
  SKILLS.map((skill) => [skill.name, skill]),
)

/** 未知工具名（后端 registry 与前端常量漂移时）回退为原名展示。 */
export function skillLabel(name: string): string {
  return SKILL_MAP[name]?.label ?? name
}

export function skillTooltip(name: string): string {
  const meta = SKILL_MAP[name]
  if (meta === undefined) return `${name}（未在前端工具清单中登记）`
  return meta.note !== undefined ? `${meta.description}\n${meta.note}` : meta.description
}
