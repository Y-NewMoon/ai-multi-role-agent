import { skillLabel, skillTooltip } from '../constants/skills'

interface Props {
  skills: string[]
  /** 为空时是否显示「无工具」占位（列表里默认不显示，避免噪音）。 */
  showEmpty?: boolean
}

/** 角色已启用工具的紧凑徽标；tooltip 说明每个工具的限制。 */
export default function SkillBadges({ skills, showEmpty = false }: Props) {
  if (skills.length === 0) {
    return showEmpty ? (
      <span className="skill-badge muted" title="未启用任何工具">
        无工具
      </span>
    ) : null
  }

  return (
    <span className="skill-badges">
      {skills.map((name) => (
        <span key={name} className="skill-badge" title={skillTooltip(name)}>
          {skillLabel(name)}
        </span>
      ))}
    </span>
  )
}
